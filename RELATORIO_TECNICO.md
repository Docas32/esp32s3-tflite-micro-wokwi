# Análise Técnica e Documentação: TensorFlow Lite Micro no ESP32-S3

Este relatório documenta a arquitetura, o fluxo de dados e as particularidades de engenharia observadas na implantação e simulação do firmware **hello_world** (aproximação da função seno) utilizando **TensorFlow Lite Micro (TFLite Micro)**, **ESP-IDF v6.1** e o simulador **Wokwi** com o microcontrolador **ESP32-S3**.

---

## 1. Visão Geral do Pipeline de TinyML

O desenvolvimento de aplicações de Machine Learning para microcontroladores segue uma esteira rigorosa de miniaturização para converter modelos abstratos de alto nível em binários executáveis em hardware embarcado:

1. **Treinamento (Python/TensorFlow/Keras)**:
   - Uma rede neural do tipo MLP (*Multi-Layer Perceptron*) rasa é treinada em ambiente convencional para aproximar a curva $y = \sin(x)$ no intervalo $[0, 2\pi]$.
   - A arquitetura típica conta com 1 entrada escalar ($x$), 1 a 2 camadas ocultas densas (*Fully Connected*) com funções de ativação não lineares (como ReLU) e 1 saída escalar ($y$).

2. **Conversão e Quantização pós-treinamento (`TFLiteConverter`)**:
   - O modelo Keras (`.h5` ou *SavedModel*) é exportado para o formato FlatBuffer do TensorFlow Lite (`.tflite`).
   - Aplica-se a quantização completa para inteiros de 8 bits (**int8 full integer quantization**). Esse processo mapeia pesos e ativações de ponto flutuante (`float32`) para inteiros com sinal (`int8`) através de parâmetros de escala (*scale*) e deslocamento (*zero point*):
     $$q = \text{round}\left(\frac{r}{S}\right) + Z$$
   - **Benefício**: Reduz o tamanho do modelo em ~75% e viabiliza execução ultra rápida via aritmética inteira, dispensando coprocessadores matemáticos pesados ou FPU em ponto flutuante duplo.

3. **Serialização em Código C (`xxd -i`)**:
   - Microcontroladores frequentemente não dispõem de sistema de arquivos embarcado (ou o acesso a SPIFFS/FATFS adiciona overhead de I/O desnecessário).
   - O comando de linha de comando `xxd -i model.tflite > model.cc` converte o arquivo binário FlatBuffer em um array C estático contíguo (`const unsigned char g_model[]`).
   - Isso permite que o compilador (`xtensa-esp32s3-elf-gcc`) posicione o modelo diretamente na memória flash de leitura (seção `.rodata`), consumindo zero de memória RAM para armazenamento dos pesos.

---

## 2. Papel dos Arquivos Principais no Ciclo de Vida do Projeto

O projeto adota uma arquitetura em camadas desacoplada:

| Arquivo | Papel e Responsabilidade |
| :--- | :--- |
| **`main/main.cc`** | Ponto de entrada padrão da aplicação no FreeRTOS (`app_main`). Inicializa o framework chamando `setup()` e executa um laço contínuo (`while (true)`) que chama `loop()` com cadência de 500 ms (`vTaskDelay`). |
| **`main/main_functions.cc / .h`** | Orquestrador principal da inferência no estilo Arduino (`setup()` / `loop()`). Contém a lógica de instanciação da *tensor arena*, registro de operadores, quantização de entrada, execução da inferência e desquantização da saída. |
| **`main/constants.cc / .h`** | Centraliza constantes físicas e de simulação: `kXrange = 2 * PI` (faixa do domínio matemático) e `kInferencesPerCycle = 20` (amostragem angular em cada período da onda). |
| **`main/model.cc / .h`** | Exporta a estrutura binária do modelo (`g_model` e `g_model_len`) gerada a partir do FlatBuffer do TensorFlow Lite. |
| **`main/output_handler.cc / .h`** | Camada de abstração de saída de dados (`HandleOutput`). Isola o mecanismo de exibição (neste caso, envio serial formatado via `MicroPrintf`), facilitando portar o mesmo código para LEDs PWM ou displays sem alterar o núcleo do algoritmo. |

### Fluxo de Execução (`setup()` e `loop()`):
- **`setup()`**:
  1. Carrega o modelo com `tflite::GetModel(g_model)`.
  2. Registra apenas os kernels estritamente necessários via `tflite::MicroMutableOpResolver<1>` (no caso, apenas `AddFullyConnected()`).
  3. Instancia o `tflite::MicroInterpreter` apontando para o modelo, o resolver e a `tensor_arena`.
  4. Executa `interpreter->AllocateTensors()` para alocar a memória de trabalho necessária.
- **`loop()`**:
  1. Calcula $x$ proporcional ao índice da iteração.
  2. Quantiza $x$ para `int8_t` usando `input->params.scale` e `input->params.zero_point`.
  3. Alimenta o buffer de entrada e executa `interpreter->Invoke()`.
  4. Extrai a saída quantizada de `output->data.int8[0]`, converte de volta para ponto flutuante ($y$) e delega para `HandleOutput(x, y)`.

---

## 3. Inferência em Tempo de Execução e Alocação Estática de Memória

Diferente de runtimes convencionais de desktop/servidor (como TensorFlow, PyTorch ou ONNX Runtime), o **TensorFlow Lite Micro** opera sob severas restrições de tempo real e memória:

1. **Ausência de Alocação Dinâmica (`malloc`/`free`)**:
   - Em sistemas operacionais de desktop, tensores intermediários são alocados dinamicamente na *heap*.
   - Em sistemas embarcados de missão crítica, alocações dinâmicas repetitivas causam **fragmentação de memória**, levando a falhas imprevisíveis por exaustão de heap (*out-of-memory crashes*) após horas ou dias de operação contínua.
   - O TFLite Micro adota a filosofia de **alocação estática previsível**: toda a memória dinâmica que a rede precisa durante a execução é reservada antecipadamente em tempo de compilação dentro de um vetor de bytes denominado **`tensor_arena`** (`uint8_t tensor_arena[kTensorArenaSize]`).

2. **Papel da `tensor_arena` e do `MicroAllocator`**:
   - Neste exemplo, a arena foi dimensionada para **2000 bytes** (`kTensorArenaSize = 2000`).
   - Durante a chamada `AllocateTensors()`, o interpretador analisa a topologia da rede e reaproveita buffers da arena para tensores intermediários que não possuem tempo de vida concorrente (*head-to-tail allocation*), garantindo eficiência máxima no uso dos escassos bytes de SRAM.

3. **Kernel Resolver Otimizado (`MicroMutableOpResolver`)**:
   - Em vez de carregar a biblioteca inteira de operadores neurais (que aumentaria o binário em centenas de kilobytes), instancia-se um resolver com tamanho exato (`MicroMutableOpResolver<1>`) incluindo unicamente os operadores presentes na arquitetura da rede.

---

## 4. Observações de Engenharia no Ambiente ESP32-S3 + Wokwi + VS Code

Durante o processo de configuração e compilação do projeto, foram identificados e solucionados aspectos críticos para o funcionamento estável:

1. **Gestão de Dependências via IDF Component Manager**:
   - O componente `espressif/esp-tflite-micro` gerencia automaticamente a biblioteca acelerada de kernels neurais da Espressif, trazendo como dependência o **`espressif/esp-nn`**.
   - O componente `esp-nn` inclui rotinas vetoriais otimizadas que exploram as instruções estendidas (PIE - *Processor Instruction Extensions*) da arquitetura Xtensa LX7 do ESP32-S3 para multiplicação e acumulação inteira acelerada (SIMD).

2. **Roteamento de Console Serial (UART0 vs. USB-JTAG)**:
   - **Diagnóstico**: O chip ESP32-S3 possui uma controladora periférica USB Serial/JTAG nativa. Se o ESP-IDF direcionar a saída padrão de log para esse canal USB, o simulador Wokwi não detecta nenhum tráfego serial, pois monitora apenas os pinos tradicionais de hardware UART0 (GPIO43/GPIO44 ou pinos TX/RX).
   - **Solução Implementada**: Foi garantida a inclusão das diretivas no `sdkconfig.defaults`:
     ```text
     CONFIG_ESP_CONSOLE_UART_DEFAULT=y
     CONFIG_ESP_CONSOLE_UART=y
     CONFIG_ESP_CONSOLE_UART_NUM=0
     CONFIG_ESPTOOLPY_MONITOR_BAUD=115200
     ```

3. **Métricas de Ocupação de Memória**:
   - **Tamanho do Firmware (`.bin`)**: `197.904 bytes` (~193 KB), ocupando apenas **10%** da partição primária da aplicação (90% livre em partição de 1,9 MB).
   - **Memória Flash Code (`.text`)**: `88.484 bytes`.
   - **Memória Flash Data (`.rodata`)**: `50.288 bytes` (onde residem o modelo FlatBuffer e tabelas constantes).
   - **Memória RAM Interna (DIRAM)**: Apenas `47.614 bytes` utilizados de 341.760 bytes disponíveis (**13,93%** de ocupação total de RAM, restando mais de 294 KB livres para a aplicação e pilhas de rede/tarefas).

4. **Particularidades da Simulação no Wokwi**:
   - O Wokwi emula a CPU Xtensa do ESP32-S3, as interrupções de temporizador e a porta serial UART0 com precisão suficiente para validar a convergência matemática e o fluxo do modelo.
   - Por ser uma simulação estritamente voltada a firmware neste circuito, qualquer componente visual periférico (como um LED RGB endereçável WS2812 frequentemente presente nas placas físicas ESP32-S3-DevKitC-1) não emite luz a menos que seja explicitamente modelado no `diagram.json` e acionado no firmware via driver RMT/LEDC.

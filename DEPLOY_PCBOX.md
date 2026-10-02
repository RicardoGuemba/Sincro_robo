# Implantação posterior no PCBOX

Este roteiro deve ser executado somente depois de publicar uma versão identificada no GitHub. Ele não faz parte da validação local atual.

## 1. Pré-requisitos de campo

- PCBOX Lubuntu x86_64 com CPython 3.12.
- SentechSDK 1.2.3 instalado e `/opt/sentech/.stprofile` presente.
- Wheel `stapipy-1.2.3-cp312-cp312-linux_x86_64.whl` disponível localmente.
- Driver NVIDIA/CUDA compatível com a instalação do PyTorch/RF-DETR.
- IP real do Omron NX102 confirmado.
- `RobFrom_Coord_CurrBase_Tool` publicado via EtherNet/IP e com tipo/unidades conferidos no Sysmac.
- Supervisório, StViewer e qualquer outro cliente StApi parados antes de abrir a câmera.

## 2. Instalação a partir do GitHub

Clone em um diretório novo. Não substitua `/opt/sincro_robo`: essa pasta é a coleta que já roda na célula.

```bash
git lfs install
git clone https://github.com/RicardoGuemba/Sincro_Calibra_robo.git /opt/sincro_calibra_robo
cd /opt/sincro_calibra_robo
git lfs pull
export STAPIPY_WHEEL=/caminho/stapipy-1.2.3-cp312-cp312-linux_x86_64.whl
./scripts/install_pcbox.sh
```

O `git lfs pull` baixa o checkpoint de 128 MB. Sem ele, o arquivo fica como ponteiro de texto e a instalação para. O bundle esperado é:

```text
buddmeyer_rfdetr_seg__seg_small__20260915_183353/
├── checkpoint_best_total.pth
├── config.json
├── hardware_env.json
├── manifest.json
└── metrics.csv
```

Esta versão usa `pick_offset_local_mm = [55.0, 0.0]`. Confirme esse vetor no ferramental antes de gravar pares do molde.

## 3. Preflight sem escrita no CLP

```bash
export SINCRO_PLC_IP=<IP_CONFIRMADO_DO_NX102>
set +u
source /opt/sentech/.stprofile
set -u
source .venv/bin/activate
python scripts/preflight.py --config config/pcbox.json --check-network
```

O preflight verifica ambiente, módulos, checkpoint e apenas a abertura TCP para a porta 44818. Ele não escreve tags e não abre a câmera.

## 4. Primeira execução assistida

```bash
export SINCRO_PLC_IP=<IP_CONFIRMADO_DO_NX102>
./scripts/run_pcbox.sh
```

Abra `http://<IP_DO_PCBOX>:8080`. Confirme visualmente:

1. cores e geometria do frame contra o StViewer;
2. uma única máscara `sku` e threshold inicial 0,3;
3. centroide e eixo sobre a máscara na imagem original;
4. θv horário em `[0°, 180°)` para 0°, 90° e orientações próximas de 179°;
5. leitura de X/Y/Z/Rx/Ry/Rz contra o watch do Sysmac;
6. tipo do array e unidades reais de posição/orientação;
7. sentido local do offset configurado, `[55.0, 0.0]`, contra o ferramental;
8. câmera liberada corretamente no encerramento.

## 5. Itens que não podem ser assumidos

- Não copiar IP do CLP de outro projeto.
- Não criar tags adicionais.
- Não escrever em `RobFrom_Coord_CurrBase_Tool`.
- Não rotular variação de pose como permissivo de segurança ou `robot_stopped`.
- Não reduzir automaticamente as tolerâncias de 20 mm e 5°.

## 6. Serviço opcional

O arquivo `deploy/systemd/sincro-robo.service` é um modelo. Antes de habilitá-lo, crie usuário/grupo dedicados, ajuste `WorkingDirectory` e grave somente o IP confirmado em `/etc/sincro-robo.env`:

```text
SINCRO_PLC_IP=192.168.x.x
```

Não habilite início automático antes da validação de exclusão da câmera com o supervisório existente.


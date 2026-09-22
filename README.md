# SINCRO_ROBO

Aplicação web local para calibração manual do relacionamento entre a câmera Omron Sentech, a segmentação RF-DETR e a pose de um robô lida no Omron NX102 via CIP.

A implementação segue o **PRD + SDD v0.4** e o `Backup/PLAYBOOK_STAPI_CIP.md`. Ela não comanda movimentos, não executa handshake e não verifica permissivos de segurança. A única escrita CIP é o toggle de `VisionCtrl_Heartbeat`; `RobFrom_Coord_CurrBase_Tool` continua só leitura.

O perfil operacional é o PCBOX (`config/pcbox.json`). `config/simulator.json` permanece só para testes automatizados.

## O que está implementado

- adaptador StApiPy com ciclo de vida na mesma thread; frame limitado a 960 px no lado maior (960×720, 4:3);
- RF-DETR `RFDETRSegSmall` no bundle `NEW_buddmeyer_…`; inferência em frame inteiro;
- centroide C, eixo `minAreaRect`, θ norte em `[0°, 180°]` (o vetor nunca aponta para sul);
- VCPn = C + 55 mm no sentido do vetor (`mm_per_px` 1,026 no plano Z = 200 mm);
- ROI visual `[66, 64, 841, 615]` (retângulo, bússola N/L, quadrante Q); não escolhe extrema nem corta o RF-DETR;
- overlay: máscara, ROI, bússola, C, seta C→VCPn, VCPn e HUD;
- HMI de visão: C (CX/CY) → vetor (θ) → VCPn (X/Y) + Q;
- offset de garra no frame da ferramenta `[0,0, 55,0]` mm (Y local), distinto do offset visual C→VCPn;
- leitura CIP de `RobFrom_Coord_CurrBase_Tool[0..5]`;
- watchdog CIP: toggle `VisionCtrl_Heartbeat` a cada 1 s, eco `PlcCtrl_HeartBeat`; morto se o eco não mudar em 3 s ou se a I/O CIP falhar;
- captura 2/2 bloqueada até o primeiro flip do eco (e de novo após reconexão); último XYZ permanece com alarme `CIP SEM ECO`;
- chip **CLP ONLINE** pisca o verde no ritmo do eco; eco morto = falha vermelha;
- captura em duas etapas: visão (1/2) depois pose do robô (2/2), para o gripper poder cobrir a peça;
- RMSE e sugestão após cada ponto gravado;
- sessões com planos Z 0/200/400 mm, 5+2 pontos e expansão até 9;
- SQLite, auditoria e exportações JSON/CSV com versão e hash SHA-256 do modelo.

## Arquitetura

```text
CameraService (thread proprietária)
  ├── StApiCamera (campo) ou SyntheticCamera (testes)
  └── RFDetrSegmenter ou SyntheticSegmenter
          ↓
    MoldPoseEstimator + VisionStabilityTracker + VCPn
          ↓
       SharedState ← PLCService (thread CIP: pose + heartbeat)
          ↓
    CaptureController ── 1/2 visão congelada, 2/2 pose se o eco oscilar
          ↓
    SQLite + CalibrationEngine + Export/Audit
          ↓
        FastAPI + HMI web
```

## Executar no PCBOX

Python 3.12, SentechSDK em `/opt/sentech`, Realtec Vision parado (a câmera e o BOOL de heartbeat são exclusivos):

```bash
export SINCRO_PLC_IP=192.168.250.1   # opcional; o default do run.sh é este IP
./run.sh
```

Abra `http://<IP-do-PCBOX>:8080`. Equivalente: `./run.sh pcbox` ou `python -m sincro_robo --config config/pcbox.json`.

Fluxo de coleta:

1. crie ou abra a sessão e selecione o plano Z;
2. **CAPTURAR COORDENADAS DA VISÃO (1/2)** congela C, θ e VCPn (os gates de qualidade não bloqueiam);
3. posicione o robô; a câmera pode perder o objeto;
4. com o chip CLP em online e o LED a piscar no eco, **CAPTURAR COORDENADAS DO ROBÔ (2/2)** grava o ponto;
5. se o eco parar, o painel mantém o último XYZ, mostra `CIP SEM ECO` e o 2/2 fica bloqueado; **Descartar visão congelada** volta ao 1/2;
6. acompanhe RMSE, região sugerida e métricas;
7. exporte JSON e CSV ao final.

Implantação e preflight: [DEPLOY_PCBOX.md](DEPLOY_PCBOX.md).

## Testes

```bash
python -m pytest
```

A suíte cobre θ norte / VCPn, overlay, escala mm/px, captura em duas etapas, eco CIP (primeiro flip, janela de 3 s, XYZ congelado) e API. Os testes sobem com `config/simulator.json`; isso não é o modo de campo.

## Configuração

- `config/pcbox.json`: StApi + RF-DETR + CIP real (default).
- `./run.sh`: perfil `pcbox` por omissão; `./run.sh simulator` só para a suíte local.
- `SINCRO_CONFIG`: escolhe um JSON sem `--config`.
- `SINCRO_PLC_IP`: sobrescreve o IP do CLP.

Raster do frame: origem canto superior esquerdo, X+ leste, Y+ sul. Escala 0,38 mm/px em 2592×1944, multiplicada por 2,7 no downscale 960×720 → **1,026 mm/px** no plano Z = 200 mm.

## Bundle RF-DETR

O app lê `manifest.json` como referência operacional e carrega:

```text
NEW_buddmeyer_rfdetr_seg__seg_small__20260915_183353buddmeyer_rfdetr_seg__seg_small__20260915_183353/checkpoint_best_total.pth
```

Parâmetros fixados pelo bundle: `RFDETRSegSmall`, resolução 384, classe `Molde` e threshold inicial 0,3. O checkpoint registra `rfdetr_version = 1.10.1`; a dependência do perfil PCBOX é fixada nessa mesma versão. O bundle `buddmeyer_rfdetr_seg__seg_small__20260915_183353` permanece no repositório só para rollback.

## PCBOX e integração de campo

A implantação deve ocorrer pelo fluxo Git/GitHub/tag; não copie mudanças sem versionamento diretamente para o PCBOX.

Antes da v1.0 ainda precisam ser confirmados em campo:

- tipo Sysmac exato do array (REAL/LREAL) e unidades efetivas;
- convenção angular visual contra o molde físico;
- IP real do NX102 se diferente de `192.168.250.1`;
- tolerâncias finais após a campanha;
- owner/repositório do GitHub.

## Segurança operacional

- A câmera StApi aceita um cliente por device: pare o supervisório, o StViewer e o Realtec Vision.
- Dois escritores em `VisionCtrl_Heartbeat` destroem o eco; só o Sincro deve pulsar esse BOOL na calibração.
- `StApiCamera` abre, captura e encerra na mesma thread.
- Toda I/O CIP (pose, write do heartbeat, read do eco) fica na thread proprietária.
- Variação da pose é apenas diagnóstico; nunca é tratada como `robot_stopped`.
- A persistência de um `RobotPoseSnapshot` só ocorre em **CAPTURAR COORDENADAS DO ROBÔ (2/2)** com eco a oscilar.

# SDD — SINCRO_ROBO no PCBOX de Curitiba

Documento de projeto para customizar esta versão no Ubuntu do POC, com Cursor. O repositório é `https://github.com/RicardoGuemba/Sincro_robo.git`, branch `main`, diretório `/opt/sincro_robo`.

Este arquivo descreve o aplicativo que já roda a campanha óptica e a coleta do molde no mesmo processo. A customização de campo ajusta a máquina. Ela preserva a cadeia de calibração, os gates e o contrato com o robô.

## 1. Como usar este documento no Cursor

Abra o repositório já atualizado (`git pull origin main` e `git lfs pull`). No chat do Cursor, cole:

> Siga `docs/SDD_SINCRO_ROBO_PCBOX_CURITIBA.md`. Corrija somente o que impedir esta máquina Ubuntu de executar o app. Mantenha a arquitetura, os gates da campanha, a cadeia óptica e o contrato CIP. Antes de alterar um número de configuração, registre a medição que o justifica. Rode `python -m pytest` no que for possível sem a câmera. Não crie um segundo processo.

Trabalhe em commits pequenos, no mesmo `main`, e descreva no commit a medição ou o erro de campo que motivou a mudança.

## 2. O que o aplicativo faz

O SINCRO_ROBO faz a câmera e o robô apontarem para o mesmo ponto da célula.

Há dois modos no mesmo processo:

1. **Campanha óptica.** O tabuleiro cobre a lente. O app trava foco e resolução, calcula K e a distorção, ensina origem e eixos lendo a pose do robô, confere o sentido, valida três alturas e fecha um perfil com hash SHA-256.
2. **Coleta do molde.** Sem campanha ativa, a coleta atual permanece. Com a campanha na etapa do molde, o RF-DETR continua vendo o frame bruto. O centroide que entra na afim é o ponto já corrigido por K e pela matriz retificada. A afim absorve o resíduo até o centro da ferramenta e fica presa àquele perfil.

O app não move o robô, não escreve no CLP e não executa handshake de segurança.

## 3. Arquitetura

```text
thread camera-owner
  StApiCamera ou SyntheticCamera
  RFDetrSegmenter ou SyntheticSegmenter
  detecção do tabuleiro e JPEG de preview
        ↓
  SharedState  ←  thread cip-owner (leitura CIP, ou pose sintética)
        ↓
  CampaignController e CaptureController
        ↓
  SQLite + perfil óptico em data/campaigns/<id>/profile.json
        ↓
  FastAPI. O HTTP só lê estado e recebe decisão humana.
```

Arquivos que sustentam esse desenho:

| Peça | Arquivo |
| --- | --- |
| Geometria pura do tabuleiro, K, R, t, nitidez | `sincro_robo/optics.py` |
| Etapas e gates | `sincro_robo/campaign_flow.py` |
| Campanha, foco, perfil | `sincro_robo/services/campaign.py` |
| Coleta do molde e afim | `sincro_robo/services/capture.py`, `sincro_robo/calibration.py` |
| Threads e estado | `sincro_robo/services/runtime.py` |
| Câmera | `sincro_robo/adapters/camera.py` |
| Pose CIP | `sincro_robo/adapters/plc.py` |
| RF-DETR | `sincro_robo/adapters/segmenter.py` |
| Persistência | `sincro_robo/storage.py` |
| HTTP | `sincro_robo/api.py` |
| Interface | `sincro_robo/web/` |
| Perfil Ubuntu | `config/pcbox.json` |
| Instalação e preflight | `scripts/install_pcbox.sh`, `scripts/preflight.py`, `DEPLOY_PCBOX.md` |

O simulador (`config/simulator.json`) existe para o Mac. No PCBOX o processo sobe com `config/pcbox.json`.

## 4. Invariantes de execução

Estas regras vieram do uso em campo e do desenvolvimento desta versão. Uma correção no Ubuntu as mantém.

- Existe um único cliente StApi. Supervisório, StViewer e qualquer outro programa soltam a câmera antes do app abrir o device.
- Abrir, capturar e encerrar a câmera acontece na thread `camera-owner`. O HTTP não captura frame.
- O RF-DETR carrega na mesma thread, antes de um segundo cliente existir. Se o startup falhar porque a câmera foi aberta e o modelo demorou a carregar, carregue o segmentador e só então chame `camera.open()`, ainda dentro de `_vision_loop`. Não crie processo auxiliar para o modelo nem para a câmera.
- A pose lida é a tag `RobFrom_Coord_CurrBase_Tool`. A leitura é somente leitura. Não acrescente tag, não escreva heartbeat e não envie comando de movimento.
- O IP do NX102 entra por `SINCRO_PLC_IP`. Não grave IP de outro projeto dentro do código.
- A persistência de um par visão–robô ocorre depois do **GRAVAR** humano. Congelar a visão não grava o par.
- K e os coeficientes de distorção valem para o raster realmente capturado, no foco travado. O tamanho entra no fingerprint do perfil. Não fixe 960×720 nem 960×540 no cálculo de K.
- O RF-DETR recebe o frame bruto. A correção de lente, `undistortPoints` com `P = K_rect`, aplica-se ao centroide que vai para a afim.
- O export da sessão do molde permanece com `schema_version` `"1"`. Pares e CSV não ganham coluna obrigatória nova. Estado novo da campanha fica nas tabelas `calibration_campaigns` e `campaign_captures`, e o perfil óptico fica em `profile.json` com `schema_version` `"optical-1"`.
- A classe do bundle é `sku`, resolução 384, limiar inicial 0,3, pacote `rfdetr==1.10.1`, construtor `RFDETRSegSmall`. O checkpoint é `buddmeyer_rfdetr_seg__seg_small__20260915_183353/checkpoint_best_total.pth`, baixado por Git LFS. Um ponteiro de texto no lugar do `.pth` é falha de instalação: `git lfs pull`. O preflight e o `install_pcbox.sh` recusam esse ponteiro.
- `pick_offset_local_mm` sai da configuração. O valor desta versão é `[55.0, 0.0]`. Só altere depois de medir o vetor do TCP até o centro da ferramenta e registre a medição no commit.

## 5. Campanha: ordem e o que cada etapa já carrega

O avanço é bloqueado enquanto o gate da etapa não passa. Não acrescente atalho, não junte etapas e não deixe a interface marcar uma etapa como concluída sem o estado gravado.

| Etapa | O que o operador faz | O que fica gravado para a etapa seguinte |
| --- | --- | --- |
| 1. Fixar o hardware | Tabuleiro à frente da lente. Anel até a nitidez estabilizar no máximo. Travar foco, câmera e resolução. Informar o lado do quadrado medido com paquímetro. | `focus_lock.peak` e os quatro itens do checklist |
| 2. Capturar a intrínseca | Tabuleiro rígido em posições diferentes. Aceitar só imagem com 54 cantos. | Pelo menos 20 imagens e as nove regiões do campo |
| 3. Revisar a reprojeção | Calcular K. Excluir imagem acima de 1,0 px e calcular de novo. | K, K_rect e coeficientes de distorção |
| 4. Ver a imagem corrigida | Comparar antes e depois e confirmar. | A cadeia seguinte usa o ponto corrigido |
| 5. Ensinar origem e eixos | Tabuleiro parado. TCP na origem vermelha, em +X (sentido dos 9 cantos) e em +Y (sentido dos 6 cantos). | Três poses e um referencial que não seja uma reta só |
| 6. Calibrar a extrínseca | Uma imagem nova, sem mover o tabuleiro ensinado. | R, t de tabuleiro→câmera e a composição robô←câmera, usando o K já gravado |
| 7. Conferir o sentido | TCP no canto interno a um quadrado da origem, no eixo X. | O sentido ensinado precisa ter erro menor que o sentido invertido |
| 8. Validar os três planos | Imagens novas em Z = 0, 200 e 400 mm, sobre calços de altura conhecida. | MAE, RMSE e máximo em milímetros. K e R, t não são recalculados |
| 9. Afim com o molde | Retirar o tabuleiro. Coleta já conhecida, com o molde e o RF-DETR. | Sessão ligada ao hash do perfil. Centroide corrigido antes do ajuste |
| 10. Analisar os resíduos | Ler o padrão do erro. | O perfil óptico já está fechado. RANSAC só se poucos pontos estiverem muito ruins |

Constantes em `sincro_robo/optics.py`: padrão `(9, 6)`, 20 imagens, limite de reprojeção 1,0 px, área mínima do referencial 500 mm², planos `(0, 200, 400)`.

O foco, na etapa 1, mede a variância do Laplaciano num recorte fixo do tabuleiro. O cadeado só habilita quando oito amostras estão estáveis: o mínimo é pelo menos 90% do pico e o desvio é no máximo 8% da média. Depois do travamento, a referência é o pico gravado. Se a nitidez ficar abaixo de 55% desse pico por oito quadros com o tabuleiro visível, `focus_drift` bloqueia a campanha e a interface oferece recomeçar. Recomeçar marca os perfis ópticos anteriores como obsoletos. Sem tabuleiro visível, a queda não acumula.

K, os coeficientes e R, t passam sozinhos de uma etapa para a outra, no estado da campanha. O operador não copia matriz. Incluir ou excluir imagem da intrínseca apaga K e exige cálculo novo. Imagem nova na extrínseca apaga o R, t anterior. Ao entrar na etapa do molde, `_freeze_profile` grava `profile.json` e o SHA-256. Um perfil novo torna obsoletos os hashes anteriores.

A afim do molde não reaplica R, t em cada ponto. Ela recebe o centroide já corrigido pela lente. R, t permanecem no perfil e na validação dos planos do tabuleiro.

## 6. O que customizar no Ubuntu

Ajuste só quando a máquina, a lente ou o ferramental medido divergirem do que o preflight e a primeira imagem mostrarem.

| Sintoma na célula | Onde mexer | Limite |
| --- | --- | --- |
| Preflight acusa ponteiro LFS ou checkpoint ausente | `git lfs pull` no clone. Não substitua o `.pth` por outro arquivo | O SHA do bundle desta versão permanece |
| Câmera não abre, device ocupado | Parar o outro cliente StApi. `device_index` em `config/pcbox.json` se houver mais de uma câmera | Um cliente, thread `camera-owner` |
| Raster de produção diferente do simulador 960×540 | Downscale no adaptador StApi, com `INTER_AREA`, para o raster que a célula realmente usa. Registrar largura e altura no fingerprint | K antigo deixa de valer. A campanha recomeça nesse raster |
| IP ou timeout do NX102 | `SINCRO_PLC_IP` e, se preciso, `connection_timeout_s` em `config/pcbox.json` | Tag e modo somente leitura permanecem |
| Tipo Sysmac ou unidade da pose divergem do watch | Conversão na borda de `CipPoseReader`, com o tipo conferido no Sysmac escrito no commit | Não criar tag nova |
| Offset da ferramenta diferente de `[55.0, 0.0]` | `calibration.pick_offset_local_mm` em `config/pcbox.json`, com a medição no commit | O código lê a configuração. Não espalhe o número em fórmulas |
| Nitidez real oscila no pico com o anel já no máximo | Limiares de `assess_focus` só com amostra gravada da célula | O cadeado continua exigindo pico estável. Deriva continua exigindo recomeço |
| Lado do quadrado | Campo da campanha, medido com paquímetro. Default de interface 30 mm | Não fixe 30 mm dentro de `calibrateCamera` |

Fora isso, a campanha, a coleta e o modelo já estão no `main`. Não reimplemente o assistente, não troque o RF-DETR por outro detector e não abra um segundo aplicativo de calibração: a StApi aceita um cliente.

## 7. O que permanece como está

- Ordem das dez etapas e os textos de gate em `campaign_flow.py`.
- `calibrateCamera`, `solvePnP`, `undistortPoints` e a composição robô←câmera em `optics.py`.
- Validação em imagens novas. As fotos do ajuste não entram na métrica de Z.
- Sentido invertido não avança.
- Sem campanha, a coleta do molde não muda de comportamento.
- Export `schema_version` `"1"`.
- Ausência de correção de cor, flat-field, aberração cromática e modelo não linear no lugar da afim.
- RANSAC apenas quando poucos pontos estão muito ruins, na etapa 10, sem apagar os pares.
- Testes em `tests/`. Uma mudança de gate ou de geometria atualiza o teste correspondente e `python -m pytest` passa antes do commit.

## 8. Subida na máquina

```bash
cd /opt/sincro_robo
git lfs install
git pull origin main
git lfs pull
export STAPIPY_WHEEL=/caminho/stapipy-1.2.3-cp312-cp312-linux_x86_64.whl
./scripts/install_pcbox.sh
export SINCRO_PLC_IP=<IP_CONFIRMADO_DO_NX102>
set +u
source /opt/sentech/.stprofile
set -u
source .venv/bin/activate
python scripts/preflight.py --config config/pcbox.json --check-network
./scripts/run_pcbox.sh
```

Pré-requisitos que o git não leva: CPython 3.12, SentechSDK 1.2.3 com `/opt/sentech/.stprofile`, wheel `stapipy` cp312 linux_x86_64, driver NVIDIA/CUDA compatível com o PyTorch do RF-DETR.

O preflight não escreve no CLP e não abre a câmera. Ele confere Python 3.12, arquitetura x86_64, perfil Sentech, módulos, versão do `rfdetr`, checkpoint real e o TCP 44818 quando `--check-network` é usado.

Abra `http://<IP_DO_PCBOX>:8080`. Na primeira execução, confirme frame, uma máscara `sku`, centroide no frame bruto, pose X/Y/Z/Rx/Ry/Rz contra o watch do Sysmac, e o encerramento liberando a câmera.

O serviço systemd em `deploy/systemd/sincro-robo.service` fica desligado até essa conferência. O IP confirmado, se for para o serviço, vai em `/etc/sincro-robo.env` como `SINCRO_PLC_IP`.

## 9. Verificação depois de uma correção

1. `python -m pytest` no ambiente em que OpenCV e as dependências de desenvolvimento estiverem instaladas.
2. Preflight com `--check-network` verde.
3. App no ar com câmera e pose reais, sem outro cliente StApi.
4. Etapa 1: tabuleiro com 54 cantos, nitidez estável, foco só então travado.
5. Uma imagem da intrínseca aceita e, se K for recalculado, reprojeção revista antes de avançar.
6. Uma pose CIP lida na origem, sem comando de movimento saindo do app.
7. Sessão de molde aberta pela campanha mostra o perfil no config da sessão e recusa recálculo se o perfil estiver obsoleto.

Se a correção for só de IP, device ou caminho do wheel, os testes de geometria continuam válidos e o critério de aceite é o preflight mais a primeira imagem real.

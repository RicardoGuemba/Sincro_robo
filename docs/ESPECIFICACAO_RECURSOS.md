# Especificação de recursos — SINCRO_ROBO v0.4.0

Documento de baixo nível para um desenvolvedor especificar, provisionar e revisar **todos os recursos** que o sistema consome, produz ou reserva. A fonte de verdade é o código em `sincro_robo/` e os perfis `config/pcbox.json` (campo) e `config/simulator.json` (testes). Onde o campo ainda não fechou um valor, a seção [Itens em aberto](#23-itens-em-aberto-para-especificação-de-campo) marca o ponto.

Versão do pacote: `sincro_robo.__version__ = 0.4.0`. Python exigido: `>=3.12,<3.13`.

---

## 1. Fronteira do sistema

SINCRO_ROBO é um processo local de calibração manual. Relaciona a pose visual de um molde (câmera Omron Sentech + RF-DETR) com a pose lida do robô no Omron NX102 via CIP. Não comanda eixos, não faz handshake de ciclo e não avalia permissivos de segurança.

| Recurso | Direito |
| --- | --- |
| Device StApi (câmera) | exclusivo: abre, captura e fecha |
| Tag CIP `VisionCtrl_Heartbeat` | escrita exclusiva (toggle BOOL) |
| Tag CIP `PlcCtrl_HeartBeat` | leitura (eco) |
| Tag CIP `RobFrom_Coord_CurrBase_Tool` | leitura de `[0..5]` |
| Disco local (`data/`, `data/exports/`) | leitura e escrita |
| HTTP `:8080` | servidor |
| GPU / CPU para RF-DETR | inferência no processo |

Fora de fronteira (o processo não cria, não escreve e não interpreta como permissivo):

- qualquer outra tag CIP;
- movimento, velocidade, zona ou programa do robô;
- variação de pose tratada como `robot_stopped`;
- segundo cliente StApi no mesmo device;
- segundo escritor de `VisionCtrl_Heartbeat`.

---

## 2. Processo e threads

Um único processo Python. Entrada: `sincro-robo` → `sincro_robo.__main__:main`, ou `python -m sincro_robo`. O servidor é Uvicorn com `log_level=info`, host e porta vindos da CLI ou do JSON.

### 2.1 Threads

| Nome | Dono | Ciclo de vida que deve permanecer nela |
| --- | --- | --- |
| thread do Uvicorn (event loop) | FastAPI | só lê `SharedState` e chama `CaptureController` |
| `camera-owner` | `ApplicationRuntime._vision_loop` | `segmenter.load()`, `StApiCamera.open/grab/close`, inferência, overlay, JPEG |
| `cip-owner` | `ApplicationRuntime._plc_loop` | `CipPoseReader.connect/read/write/close`, toggle e watchdog |

`StApiCamera` e `CipPoseReader` guardam `threading.get_ident()` na abertura. Qualquer chamada de outra thread levanta `RuntimeError`. As duas threads são daemon e são encerradas por `threading.Event` com `join(timeout=5)`.

Ordem obrigatória em `camera-owner`:

1. `segmenter.load()` (RF-DETR faz um predict dummy de `resolution×resolution`);
2. só então `camera.open()` e o loop de grab.

Carregar o modelo com o stream StApi já aberto estoura `RetrieveBuffer` e `/api/frame` permanece 503.

### 2.2 Locks

| Lock | Protege |
| --- | --- |
| `SharedState._lock` (`RLock`) | visão, pose, JPEG, status de hardware, eco |
| `CaptureController._lock` (`RLock`) | sessão ativa, plano ativo, visão congelada, último feedback |
| `Storage._lock` (`RLock`) | escritas SQLite |

Leituras HTTP de frame pegam o JPEG já codificado sob o lock. Não há fila de frames: o último JPEG substitui o anterior.

### 2.3 Estados de hardware publicados

Cada serviço (`camera`, `model`, `plc`) publica `status` ∈ `{starting, online, error, offline}` e `error: str | null`.

- `starting`: valor inicial antes do loop.
- `online`: último ciclo sem exceção. No CLP CIP, `online` exige watchdog saudável (seção 6).
- `error`: exceção no ciclo, ou eco morto (`CIP SEM ECO`).
- `offline`: `stop()` pedido e thread saindo. Se a thread de visão morre sem `stop`, câmera fica `error`.

---

## 3. Rede

| Socket | Direção | Perfil campo | Perfil simulador |
| --- | --- | --- | --- |
| TCP `app.host:app.port` | escuta HTTP | `0.0.0.0:8080` | `127.0.0.1:8080` |
| TCP `plc.ip:44818` | cliente CIP explícito (EtherNet/IP) | `192.168.250.1:44818` | não abre |

CLI `--host` e `--port` sobrescrevem o JSON. `SINCRO_PLC_IP` sobrescreve `plc.ip` depois do merge.

Timeouts de rede:

| Parâmetro | Valor | Uso |
| --- | --- | --- |
| `plc.connection_timeout_s` | 10,0 s | `NSeries.connect_explicit` |
| `plc.poll_interval_s` | 0,2 s | espera entre ciclos CIP bem-sucedidos |
| backoff após falha CIP | `min(3,0, max(0,5, poll×5))` = 1,0 s | espera antes de reconectar |
| preflight TCP | 3 s | `socket.create_connection` só com `--check-network` |
| `camera.fetch_timeout_ms` | 400 ms | `datastream.retrieve_buffer` |

O preflight **não** abre a câmera e **não** escreve tag. Só testa reachability da porta 44818.

Não há TLS, autenticação HTTP nem bind em outra porta. Documentação OpenAPI fica em `GET /api/docs`. ReDoc está desligado.

---

## 4. Câmera (StApi)

Recurso exclusivo: um device Sentech por processo. Pare supervisório, StViewer e Realtec Vision antes de abrir.

### 4.1 Ambiente nativo

`apply_sentech_environment()` só age se existir `/opt/sentech/lib`. Define, antes de `import stapipy`:

| Variável | Prefixo acrescentado |
| --- | --- |
| `STAPI_ROOT_PATH` | `/opt/sentech` |
| `LD_LIBRARY_PATH` | `/opt/sentech/lib`, `/opt/sentech/lib/GenICam` (se existirem) |
| `LIBRARY_PATH` | idem |
| `GENICAM_GENTL64_PATH` | `/opt/sentech/lib` |

`run.sh` e `scripts/run_pcbox.sh` exigem `/opt/sentech/.stprofile` e fazem `source` dele no perfil `pcbox`. SDK esperado em campo: SentechSDK 1.2.3, wheel `stapipy-1.2.3-cp312-cp312-linux_x86_64.whl` (variável `STAPIPY_WHEEL` no instalador).

### 4.2 Abertura

1. `stapipy.initialize()`
2. `create_system()`
3. `device_index <= 0` → `create_first_device()`; senão `create_first_interface().create_device_by_index(device_index)`
4. `create_datastream()` → `start_acquisition()` → `device.acquisition_start()`

`device_index` de campo: `0`.

Fechamento, na mesma thread: `acquisition_stop`, `stop_acquisition`, `st.terminate()`. Falha no `open` chama `close()` antes de propagar.

### 4.3 Frame

| Grandeza | Valor |
| --- | --- |
| Sensor de referência | 2592×1944 (4:3) |
| Saída após limite | lado maior ≤ `max_dimension` = 960 → **960×720** |
| Interpolação | `cv2.INTER_AREA` |
| Upscale | nunca |
| Frame já ≤ 960 | devolvido sem cópia de escala |
| Pixel format | Bayer (RG/GR/GB/BG via `EStPixelColorFilter`, fallback GRBG), Mono, ou >8 bit com shift `valid_bits-8` |
| Layout de saída | `uint8` BGR contíguo, origem no canto superior esquerdo |
| Cadência alvo | `target_fps` = 8,0 → período 125 ms; piso do período = 0,5 fps |
| JPEG publicado | qualidade 86, `Cache-Control: no-store` |

Raster de imagem: **X+ leste, Y+ sul**.

Buffer sem `is_image_present` ou `retrieve_buffer` nulo → `TimeoutError`. O loop limpa a observação, marca câmera `error` e espera um período.

---

## 5. Modelo RF-DETR

### 5.1 Bundle ativo

Diretório (nome literal, duplicado):

```text
NEW_buddmeyer_rfdetr_seg__seg_small__20260915_183353buddmeyer_rfdetr_seg__seg_small__20260915_183353/
├── checkpoint_best_total.pth    # único arquivo carregado pelo runtime
├── manifest.json                # referência operacional; o runtime não o lê
├── config.json                  # metadados de treino; o runtime não o lê
├── hardware_env.json
└── metrics.csv
```

O diretório `buddmeyer_rfdetr_seg__seg_small__20260915_183353/` fica no repositório só para rollback. O runtime não o carrega.

| Campo do manifesto | Valor fixado no app |
| --- | --- |
| `ctor` | `RFDETRSegSmall` |
| `resolution` | 384 |
| `class_names` | `["Molde"]` |
| `default_threshold` | 0,3 |
| `variant` | `seg_small` |
| `task` | `instance_segmentation` |
| pacote `rfdetr` | `==1.10.1` (o checkpoint registra a mesma versão) |

`RFDetrSegmenter.load()` instancia `RFDETRSegSmall(pretrain_weights=checkpoint, resolution=384)` e aquece com uma imagem preta 384×384. `device` no JSON é `"auto"`; o construtor atual **não** repassa esse campo ao `RFDETRSegSmall`.

Inferência: frame BGR inteiro convertido para RGB PIL, `predict(image, threshold=0.3)`. Máscaras em `detections.mask` (bool). Se `ndim == 2`, vira um lote de uma máscara. Confiança vem de `detections.confidence`; nome de classe de `detections.data["class_name"]` ou, na falta, `class_name` da config.

O loop escolhe a máscara de **maior confiança** para geometria e overlay, mas `instance_count` é o total de máscaras acima do threshold. O gate `single_instance` exige contagem == 1. O ROI **não** filtra detecções e **não** recorta o tensor de entrada.

Identidade do modelo gravada na sessão: SHA-256 do ficheiro de checkpoint (`missing` se o ficheiro não existe) e `git rev-parse --short HEAD` (`unversioned` se o git falhar, timeout 3 s).

### 5.2 Segmentador sintético

Só com `model.provider = synthetic`. Limiar de cor no BGR do simulador: `G>150`, `R>130`, `B<120`, componentes ≥ 200 px, confiança fixa 0,995, classe `"sku"`.

---

## 6. CLP CIP

Biblioteca: `aphyt` (`omron.n_series.NSeries.connect_explicit`). Provider de campo: `cip`. Provider `synthetic` não abre socket e devolve uma pose afim determinística da visão (só testes).

### 6.1 Tags

| Tag | Acesso | Tipo esperado | Semântica |
| --- | --- | --- | --- |
| `RobFrom_Coord_CurrBase_Tool` | leitura | array numérico, 6 elementos | `[X, Y, Z, Rx, Ry, Rz]` da ferramenta na base corrente |
| `RobFrom_Coord_CurrBase_Tool[0]` … `[5]` | leitura alternativa | escalar numérico | usado se a leitura do array não devolver `list/tuple` com `len >= 6` |
| `VisionCtrl_Heartbeat` | **escrita** | BOOL | toggle do SINCRO |
| `PlcCtrl_HeartBeat` | leitura | BOOL (ou string `1/true/on/yes`, ou `bytes` com primeiro byte) | eco produzido pelo CLP |

Unidades de posição assumidas: milímetros. `plc.angle_unit`: `degrees` (campo) ou `radians` (convertido com `math.degrees` só em Rx/Ry/Rz). O tipo Sysmac exato (REAL vs LREAL) e a unidade efetiva **não estão fechados no código**; ver seção 23.

A pose lida nasce com `fresh=True` dentro do reader. Quem publica frescura real é `_publish_robot`: `fresh` passa a ser o resultado do watchdog. Pose antiga permanece no estado com `fresh=False` quando a leitura nova falha ou o eco morre. Não há outro critério de “pose parada”.

`plc.freshness_timeout_s` (default 1,5 s) existe no `DEFAULT_CONFIG` e **não é lido** por nenhum módulo.

### 6.2 Heartbeat

`HeartbeatToggle(interval_s=1,0)`:

- o primeiro `next_value` é devido imediatamente;
- inverte o BOOL e devolve o novo valor;
- devolve `None` se ainda não passou `interval_s` desde o último toggle (não escreve).

`EchoWatchdog(lost_after_s=3,0)`:

1. o primeiro valor de eco é memorizado e **não** conta como aresta;
2. a primeira mudança de valor arma `_seen_edge` e o relógio;
3. `healthy` é verdadeiro só se já houve aresta e `(now - última mudança) < 3,0 s`;
4. `mark_io_error()` (qualquer exceção CIP) força não saudável até um `observe_echo` posterior;
5. `connect()` bem-sucedido chama `reset_session()` no watchdog e `reset()` no toggle.

Alarme publicado quando não saudável: string fixa `CIP SEM ECO`. Com eco saudável, `plc.echo` é o último BOOL observado; caso contrário `null`.

Efeito na captura: o passo 2/2 exige `robot.fresh`. Sem a primeira aresta, ou com eco parado, ou com I/O em falha, a pose não é gravada. O último XYZ continua visível.

Reconexão: `connected=False`, `reader.close()`, publicação com pose `None` (o snapshot anterior, se existir, fica `fresh=False`), espera de backoff, novo `connect`.

---

## 7. Sistema de ficheiros

Raiz: diretório do repositório (`PROJECT_ROOT`). Caminhos relativos em `app.data_dir`, `app.database`, `app.exports_dir`, `model.bundle_dir` e `model.checkpoint` são resolvidos contra essa raiz.

| Caminho | Papel | Quem cria |
| --- | --- | --- |
| `data/` | diretório de estado | `Storage` (pai do sqlite) |
| `data/sincro_robo.sqlite3` | base SQLite, `journal_mode=WAL`, `foreign_keys=ON`, timeout 10 s | `Storage._initialize` |
| `data/exports/` | JSON e CSV exportados | `create_app` e `export_session` |
| `config/pcbox.json` | override de campo | versionado |
| `config/simulator.json` | override de teste | versionado |
| `sincro_robo/web/` | HMI estática (`index.html`, `app.js`, `styles.css`) | pacote |
| `/opt/sincro_robo` | instalação de campo prevista | operador |
| `/etc/sincro-robo.env` | `EnvironmentFile` do systemd; deve conter `SINCRO_PLC_IP` | operador |
| `/opt/sentech/` | SDK | instalador Sentech |

Ficheiros de export (por sessão, prefixo dos 8 primeiros caracteres do UUID):

- `sincro_robo_<8hex>.json`
- `sincro_robo_<8hex>_pontos.csv`

Serviço systemd modelo: `deploy/systemd/sincro-robo.service`.

- `User`/`Group`: `sincro-robo`
- `WorkingDirectory`: `/opt/sincro_robo`
- `ExecStart`: `/opt/sincro_robo/scripts/run_pcbox.sh`
- `Restart=on-failure`, `RestartSec=5`
- `NoNewPrivileges=true`, `PrivateTmp=true`
- `After`/`Wants`: `network-online.target`

Não habilitar o unit antes de validar que nenhum outro processo segura a câmera.

---

## 8. Variáveis de ambiente

| Variável | Efeito | Default se ausente |
| --- | --- | --- |
| `SINCRO_CONFIG` | caminho do JSON se `--config` não for passado | `config/pcbox.json` |
| `SINCRO_PLC_IP` | substitui `plc.ip` | no `run.sh` / `run_pcbox.sh` de perfil pcbox: `192.168.250.1` |
| `STAPIPY_WHEEL` | caminho do wheel no `scripts/install_pcbox.sh` | obrigatória nesse script |
| `STAPI_ROOT_PATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `GENICAM_GENTL64_PATH` | injetadas por `apply_sentech_environment` e pelo `.stprofile` | — |

Não há ficheiro `.env` lido pelo processo Python. O systemd lê `/etc/sincro-robo.env`.

---

## 9. Configuração

Merge: `DEFAULT_CONFIG` profundo + JSON. Chave ausente no JSON herda o default. `config["_config_path"]` guarda o caminho absoluto do ficheiro carregado.

### 9.1 `app`

| Chave | Tipo | Default | Campo (`pcbox.json`) |
| --- | --- | --- | --- |
| `name` | str | `SINCRO_ROBO` | herdado |
| `host` | str | `127.0.0.1` | `0.0.0.0` |
| `port` | int | `8080` | `8080` |
| `data_dir` | path | `data` | herdado |
| `database` | path | `data/sincro_robo.sqlite3` | herdado |
| `exports_dir` | path | `data/exports` | herdado |

### 9.2 `camera`

| Chave | Tipo | Default | Campo |
| --- | --- | --- | --- |
| `provider` | `synthetic` \| `stapi` | `synthetic` | `stapi` |
| `device_index` | int | `0` | `0` |
| `fetch_timeout_ms` | int | `400` | `400` |
| `target_fps` | float | `8.0` | `8.0` |
| `width` | int | `960` | só simulador |
| `height` | int | `720` | só simulador |
| `max_dimension` | int | `960` | `960` |

### 9.3 `model`

| Chave | Tipo | Default | Campo |
| --- | --- | --- | --- |
| `provider` | `synthetic` \| `rfdetr` | `synthetic` | `rfdetr` |
| `bundle_dir` | path | bundle `NEW_buddmeyer_…` | igual |
| `checkpoint` | path | `…/checkpoint_best_total.pth` | igual |
| `threshold` | float | `0.3` | `0.3` |
| `resolution` | int | `384` | `384` |
| `class_name` | str | `Molde` | `Molde` |
| `device` | str | `auto` | `auto` (não repassado ao ctor) |
| `rfdetr_version` | str | `1.10.1` | `1.10.1` (preflight compara com o pacote instalado) |

### 9.4 `plc`

| Chave | Tipo | Default | Campo |
| --- | --- | --- | --- |
| `provider` | `synthetic` \| `cip` | `synthetic` | `cip` |
| `ip` | str | `""` (CIP recusa vazio) | `192.168.250.1` |
| `port` | int | `44818` | herdado |
| `connection_timeout_s` | float | `10.0` | `10.0` |
| `poll_interval_s` | float | `0.2` | `0.2` |
| `freshness_timeout_s` | float | `1.5` | **não consumido** |
| `pose_tag` | str | `RobFrom_Coord_CurrBase_Tool` | igual |
| `heartbeat_write_tag` | str | `VisionCtrl_Heartbeat` | igual |
| `heartbeat_echo_tag` | str | `PlcCtrl_HeartBeat` | igual |
| `heartbeat_interval_s` | float | `1.0` | `1.0` |
| `heartbeat_lost_after_s` | float | `3.0` | `3.0` |
| `angle_unit` | `degrees` \| `radians` | `degrees` | `degrees` |

### 9.5 `vision_quality`

Gates de diagnóstico. Nenhum deles bloqueia `POST /api/capture`.

| Chave | Default | Significado |
| --- | --- | --- |
| `min_confidence` | `0.3` | gate `confidence` |
| `min_axis_quality` | `1.35` | razão eixo maior/menor |
| `border_margin_px` | `3` | máscara tocando esta margem ⇒ `mask_cut` |
| `min_mask_area_ratio` | `0.002` | fração de pixels da máscara no frame |
| `max_mask_area_ratio` | `0.85` | idem, teto |
| `stability_samples` | `6` | tamanho da janela (mínimo efetivo 2) |
| `max_jitter_x_px` | `2.5` | σ de CX nativo |
| `max_jitter_y_px` | `2.5` | σ de CY nativo |
| `max_jitter_angle_deg` | `1.5` | RMS do delta de heading na janela |

Se qualquer gate falha, o tracker zera o histórico e publica `stable=false`. Estável exige janela cheia e os três σ dentro do limite. Jitter angular usa `signed_heading_delta` (período 360°, resultado em (−180, 180]).

### 9.6 `pixel_reference`

| Chave | Default e campo |
| --- | --- |
| `source_width` | `2592` |
| `source_height` | `1944` |
| `destination_width` | `960` |
| `destination_height` | `720` |

Escala configurada: `destination/source` → `960/2592` e `720/1944` = `10/27` ≈ 0,370370.

`reference_scale_factors(frame)`:

- frame já é o destino (960×720) → `(1, 1)`; centroide permanece em pixels do frame;
- frame é exatamente a fonte (2592×1944) → aplica `destination/source` uma vez;
- qualquer outra resolução → identidade.

No perfil de campo o grab já desce para 960×720, logo a observação gravada está no frame 960×720 (`scale_x = scale_y = 1`).

### 9.7 `vision_reference`

| Chave | Default | Campo |
| --- | --- | --- |
| `vcp_offset_mm` | `55.0` | herdado |
| `mm_per_px` | `1.026` | `1.026` |
| `roi_enabled` | `true` | herdado |
| `roi_px` | `[66, 64, 841, 615]` | igual, xywh no frame 960×720 |

Escala física declarada: 0,38 mm/px em 2592×1944, multiplicada por 2,7 no downscale para 960×720 → **1,026 mm/px** no plano Z = 200 mm. O código aplica `mm_per_px` de forma uniforme; não há tabela por plano Z.

`roi_px` é só desenho e quadrante. Não escolhe a instância e não corta a máscara.

### 9.8 `calibration`

| Chave | Default | Campo | Papel |
| --- | --- | --- | --- |
| `default_planes_mm` | `[0, 200, 400]` | herdado | sugestão de UI; a sessão grava os planos pedidos no POST |
| `default_points_per_plane` | `7` | herdado | alvo inicial por plano |
| `max_points_per_plane` | `9` | herdado | teto da expansão |
| `adjustment_points_per_plane` | `5` | herdado | documentado; a sequência real está em `POINT_REGIONS` |
| `validation_points_per_plane` | `2` | herdado | idem |
| `pick_offset_local_mm` | `[0.0, 55.0]` | `[0.0, 55.0]` | offset da garra no frame da ferramenta, mm `[dx, dy]` |
| `xy_tolerance_mm` | `20.0` | herdado | limite provisório de residual XY |
| `angular_tolerance_deg` | `5.0` | herdado | limite provisório de erro angular |
| `duplicate_distance_px` | `12.0` | herdado | distância em pixels de referência (`vision.x/y`) |
| `duplicate_angle_deg` | `3.0` | herdado | \|delta de eixo\| abaixo disto, junto com a distância, marca duplicata |
| `plan_z_tolerance_mm` | `20.0` | herdado | \|`robot.z − plano`\| para o gate `plan_z` |
| `target_region_radius_norm` | `0.24` | herdado | raio normalizado no frame para o gate `region` |

O default morto de `robot_geometric_center(..., (57.5, 0.0))` **não** é o valor de campo. Quem chama passa sempre `calibration.pick_offset_local_mm`.

---

## 10. Tipos de domínio

Timestamps: `datetime.now(timezone.utc).isoformat(timespec="milliseconds")`.

### 10.1 `VisionObservation`

| Campo | Unidade / domínio | Origem |
| --- | --- | --- |
| `timestamp` | ISO-8601 ms UTC | construção |
| `x`, `y` | px no frame de referência | centroide nativo × escala |
| `angle_deg` | graus, heading norte | seção 11 |
| `confidence` | 0–1 | melhor máscara |
| `axis_quality` | razão adimensional ≥ 1 | `major/minor` do `minAreaRect` |
| `mask_area_ratio` | 0–1 | média da máscara binária |
| `mask_cut` | bool | toque na margem |
| `instance_count` | int | máscaras acima do threshold |
| `stable` | bool | tracker |
| `sigma_x`, `sigma_y` | px | desvio-padrão da janela, coordenadas nativas |
| `sigma_angle_deg` | graus | RMS dos deltas de heading |
| `frame_width`, `frame_height` | px | shape do frame após downscale |
| `gates` | mapa de 5 bool | seção 13 |
| `native_x`, `native_y` | px no frame da câmera | centroide antes da escala |
| `scale_x`, `scale_y` | adimensional | `reference_scale_factors` |
| `reference_width`, `reference_height` | px | destino configurado |
| `vcpn_x`, `vcpn_y` | px de referência | VCPn escalado |
| `native_vcpn_x`, `native_vcpn_y` | px nativos | VCPn no frame |
| `roi_quadrant` | `NE` \| `NO` \| `SE` \| `SO` \| null | seção 11 |
| `mask_area_cm2` | cm² | `n_pixels × mm_per_px² / 100` |

`overlay_xy()` devolve nativo se existir, senão `(x, y)`. O HMI e o overlay desenham o nativo.

### 10.2 `RobotPoseSnapshot`

| Campo | Unidade |
| --- | --- |
| `timestamp` | ISO-8601 ms UTC da leitura |
| `x`, `y`, `z` | mm (assumido) |
| `rx`, `ry`, `rz` | graus depois da conversão |
| `fresh` | true só com eco saudável no provider CIP; no sintético o reader já devolve true |

Rx e Ry são diagnóstico de UI. Entram no par gravado e no CSV. Não entram no ajuste afim nem no offset angular (só Rz).

### 10.3 `FrozenVisionCapture`

Visão do passo 1/2, ainda sem pose. Campos: `id` (UUID4), `session_id`, `plan_z`, `point_index` (1-based), `role`, `region`, `vision`, `created_at`. Não é linha em `pairs`.

### 10.4 `CaptureCandidate`

Mesmos campos mais `robot` e `warnings` (tupla, hoje sempre vazia). `to_dict()` acrescenta `prompt = "Ponto registrado"`. O `id` do par é o `id` do congelamento.

---

## 11. Geometria

### 11.1 Centroide e eixo

`MoldPoseEstimator.estimate(mask)`:

1. centroide = média das colunas (`x`) e linhas (`y`) dos pixels verdadeiros; menos de 3 pixels → erro;
2. maior contorno externo (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`);
3. `cv2.minAreaRect`: o lado maior define o eixo; se `width < height`, soma-se 90° ao ângulo do retângulo;
4. ângulo do retângulo normalizado para **[0, 180)**;
5. se não houver contorno com ≥ 3 pontos, fallback PCA: maior autovetor da covariância, comprimentos ≈ `4·√λ`;
6. `axis_quality = major/minor` (`minor` mínimo 1 px no ramo do retângulo);
7. `mask_cut` se algum pixel da máscara cai nas `border_margin_px` bordas.

O ângulo do `minAreaRect` **não** é o heading publicado. Serve para obter o vetor unitário `(cos θ, sin θ)` no raster (Y para baixo, então esse θ é horário a partir de +X).

### 11.2 Heading norte e VCPn

`orient_north(ux, uy)`: se `uy > 0` (componente para sul), inverte o vetor. O vetor resultante nunca aponta para sul (`uy ≤ 0`).

`heading_north_deg`: `atan2(−ny, nx)` em graus, faixa **[0, 360)** reduzida na prática a **[0, 180]** porque `ny ≤ 0`.

| Heading | Sentido no raster |
| --- | --- |
| 0° | leste (+X) |
| 90° | norte (−Y de imagem) |
| 180° | oeste (−X) |

`VCPn = C + (vcp_offset_mm / mm_per_px) · (cos θ, −sin θ)`.

Com os defaults: 55 mm / 1,026 mm/px ≈ **53,606 px** ao longo do heading. Offset 0 ou `mm_per_px` 0 devolve o próprio centroide.

Este offset visual C→VCPn é independente de `pick_offset_local_mm`. Os dois valem 55 mm no perfil atual, em frames diferentes: um no plano da imagem, outro no frame da ferramenta do robô.

### 11.3 ROI e quadrante

`roi_px = [x, y, w, h] = [66, 64, 841, 615]` no frame 960×720.

Âncora da bússola: meio da aresta superior, `(x + w/2, y)`. Braços desenhados: norte (−Y) e leste (+X), rótulos `N` e `L`.

Quadrante do VCPn nativo, só se o ponto cai dentro do retângulo fechado:

| Condição (origem da imagem) | Rótulo |
| --- | --- |
| x ≥ meio e y < meio | `NE` |
| x < meio e y < meio | `NO` |
| x ≥ meio e y ≥ meio | `SE` |
| x < meio e y ≥ meio | `SO` |
| fora do retângulo | `null` |

### 11.4 Centro geométrico do robô

Frame da ferramenta, rotação plana por Rz (graus), sentido matemático:

```text
[gx]   [tcp_x]   [ cos Rz  −sin Rz ] [dx]
[gy] = [tcp_y] − [ sin Rz   cos Rz ] [dy]
```

`[dx, dy] = pick_offset_local_mm = [0, 55]` mm. O alvo do ajuste afim é `(gx, gy)`, não o TCP cru.

### 11.5 Ajuste do plano

Somente pares com `role = adjustment`, mínimo 3. Mínimos quadrados:

```text
[vision.x, vision.y, 1]  ·  A(3×2)  =  [gx, gy]
```

`rank < 3` → pontos degenerados, sem modelo. `angle_offset_deg` é a média circular de eixo (período 180°, truque do ângulo dobrado) de `robot.rz − vision.angle_deg`.

Predição: `xy' = [x, y, 1] · A`, `θ' = normalize_180(θ_visual + offset)`.

Erro XY: distância euclidiana entre `xy'` e `(gx, gy)`, em mm. Erro angular: `|signed_axis_delta(θ', rz)|` em (−90, 90], período 180°. O heading publicado é dirigido, mas o residual angular da calibração trata θ e θ+180 como o mesmo eixo.

Estatísticas (`mean`, `rms`, `max`, e `p95` só no XY): se existir par `validation`, as métricas de aceite usam só a validação; senão usam todos os pares. `passed` exige conjunto não vazio e `max` XY ≤ 20 mm e `max` angular ≤ 5°. Cada par fica `accepted` ou `suspect` com a mesma regra aplicada ao residual daquele ponto (não ao RMS).

Expansão de alvo: depois de gravar, se o plano tem ≥ 7 pontos, o modelo está pronto, `passed` é falso, a contagem atingiu o alvo corrente e o alvo &lt; 9, o alvo daquele plano sobe 1. O alvo inicial é `default_points_per_plane`.

---

## 12. Sequência de pontos

Índice 1-based. A posição na lista é `point_index − 1`. Índice &gt; 9 é recusado.

| Índice | Região | Papel | Alvo normalizado (x, y) no frame |
| --- | --- | --- | --- |
| 1 | Centro | adjustment | (0,50, 0,50) |
| 2 | Superior esquerda | adjustment | (0,22, 0,24) |
| 3 | Superior direita | adjustment | (0,78, 0,24) |
| 4 | Inferior esquerda | adjustment | (0,22, 0,76) |
| 5 | Inferior direita | adjustment | (0,78, 0,76) |
| 6 | Validação esquerda | validation | (0,14, 0,50) |
| 7 | Validação direita | validation | (0,86, 0,50) |
| 8 | Expansão superior | adjustment | (0,50, 0,14) |
| 9 | Expansão inferior | adjustment | (0,50, 0,86) |

O próximo índice é o menor inteiro ≥ 1 ainda livre em `(session, plan_z)`. Buracos são reutilizados.

Duplicata (não bloqueia o POST; só o gate `not_duplicate`): existe par no mesmo plano com distância de `(vision.x, vision.y)` &lt; 12 px **e** `|signed_axis_delta|` &lt; 3°.

Região útil (gate `region`): centroide nativo normalizado por `frame_width/height` dentro do raio 0,24 do alvo da tabela.

---

## 13. Máquina de estados da captura

Estados do controlador: `vision` (sem congelamento) e `robot` (visão congelada). Não há candidato pendente de confirmação. `POST /api/candidate/decision` com `confirm=true` responde 422. Com `confirm=false` descarta o congelamento.

```text
sem sessão ou sem plano
        │  POST /capture → 422
        ▼
passo visão (1/2)
        │  visão is None → 422 "Visão indisponível"
        │  POST /capture
        ▼
visão congelada (2/2)          evento audit capture_vision_frozen
        │  pose ausente → 422 "Pose do robô indisponível"
        │  pose.fresh == false → 422 "CIP SEM ECO"
        │  POST /capture
        ▼
par gravado, congelamento limpo  evento audit capture_confirmed
        │
        └─ evaluate + talvez expandir alvo + feedback RMSE

Descartar (confirm=false) a partir do 2/2 → volta ao 1/2, evento capture_cancelled.
Ativar outra sessão ou outro plano zera o congelamento sem gravar.
```

Pré-condição dos dois passos: sessão ativa e plano que pertença a `planes_mm` da sessão. O passo 1/2 **não** consulta gates, estabilidade, região, duplicata nem Z do robô. O passo 2/2 **não** relê a visão; usa o `VisionObservation` congelado.

Feedback após gravar:

- menos de 3 ajustes: `rmse_available=false`, mensagem com a falta, sugestão `Colete mais pontos`;
- modelo pronto e RMS XY ≤ 20 e RMS θ ≤ 5: sugestão `Adequado`;
- modelo pronto fora do limite: sugestão `Atenção — acima do limite provisório`.

O limite de aceite do plano (`passed`) usa o **máximo**, não o RMS. A sugestão da UI usa o **RMS**.

---

## 14. Persistência SQLite

### 14.1 `sessions`

| Coluna | Tipo | Notas |
| --- | --- | --- |
| `id` | TEXT PK | UUID4 |
| `name` | TEXT | 1–120 caracteres na API |
| `created_at` | TEXT | UTC |
| `started_at` | TEXT NULL | preenchido na primeira ativação |
| `status` | TEXT | `draft` na criação; `active` em `start_session`; não há estado `closed` |
| `app_version` | TEXT | `__version__` |
| `model_hash` | TEXT | SHA-256 do checkpoint |
| `git_revision` | TEXT | short HEAD |
| `config_json` | TEXT | snapshot abaixo |

Snapshot gravado na criação (não é o JSON inteiro de runtime):

```text
xy_tolerance_mm, angular_tolerance_deg, pick_offset_local_mm,
default_points_per_plane, max_points_per_plane, model_threshold,
vision_quality, source_config, pixel_reference{source_*, destination_*, scale_x, scale_y},
planes_mm, plan_targets{ "<z>": alvo }
```

`plan_targets` começa com o alvo default para cada plano. `update_plan_target` reescreve o JSON.

### 14.2 `pairs`

| Coluna | Tipo | Notas |
| --- | --- | --- |
| `id` | TEXT PK | UUID do congelamento |
| `session_id` | TEXT FK → sessions ON DELETE CASCADE | |
| `plan_z` | REAL | |
| `point_index` | INTEGER | UNIQUE (session_id, plan_z, point_index) |
| `role` | TEXT | `adjustment` \| `validation` |
| `region` | TEXT | nome da tabela da seção 12 |
| `vision_json` | TEXT | `VisionObservation` completo |
| `robot_json` | TEXT | `RobotPoseSnapshot` completo |
| `saved_at` | TEXT | UTC da gravação (não o `created_at` do candidato) |
| `status` | TEXT | nasce `accepted`; o evaluate reescreve `accepted` \| `suspect` |
| `residual_xy` | REAL NULL | mm, após o primeiro evaluate com ≥ 3 ajustes |
| `residual_angle` | REAL NULL | graus |

Índice: `idx_pairs_session_plan (session_id, plan_z, point_index)`.

### 14.3 `audit`

| Coluna | Tipo |
| --- | --- |
| `id` | INTEGER PK AUTOINCREMENT |
| `session_id` | TEXT NULL |
| `timestamp` | TEXT |
| `event` | TEXT |
| `detail_json` | TEXT |

Eventos emitidos:

| Evento | detail |
| --- | --- |
| `session_created` | `name`, `planes_mm` |
| `session_activated` | `{}` |
| `plan_activated` | `plan_z` |
| `plan_target_changed` | `plan_z`, `target` |
| `capture_vision_frozen` | `candidate_id`, `plan_z`, `point` |
| `capture_confirmed` | `pair_id`, `plan_z`, `point` |
| `capture_cancelled` | `candidate_id`, `plan_z` |
| `session_exported` | `json`, `csv` (nomes de ficheiro) |

---

## 15. Exportação

`schema_version` do JSON: `"1"`.

```text
{
  schema_version, exported_at,
  session: { ...sessão, pairs[], plans[] },
  audit: [ {timestamp, event, detail} ]
}
```

Cada item de `plans[]`: `z`, `count`, `target`, `complete` (`count >= target`), `suggestion | null` (`index`, `region`, `role`, `normalized`), `evaluation`.

`evaluation` sem modelo: `{ready:false, reason, adjustment_count}`. Com modelo: `{ready:true, model:{plan_z, affine, angle_offset_deg, adjustment_count}, metrics}`.

CSV `sincro_robo_<8>_pontos.csv`, UTF-8, cabeçalho fixo:

```text
id, plan_z, point_index, role, region, saved_at, status,
vision_x, vision_y, vision_angle_deg, confidence, axis_quality,
robot_x, robot_y, robot_z, robot_rx, robot_ry, robot_rz,
residual_xy_mm, error_angle_deg
```

URLs devolvidas pela API: `/exports/<nome>`. O mount é o diretório `exports_dir`, sem listagem adicional de segurança.

---

## 16. API HTTP

Base: o host/porta do processo. Corpo JSON. `ValueError` do controlador vira **422** `{"detail": "<mensagem>"}`. `KeyError` de sessão vira **404** `{"detail": "Sessão não encontrada"}`.

### 16.1 `GET /`

`index.html` da HMI. Fora do schema OpenAPI.

### 16.2 `GET /api/health`

```text
{ status: "ok", version, hardware: snapshot.hardware, mode: {camera, model, plc} }
```

### 16.3 `GET /api/state`

Polling da HMI a cada **700 ms**.

```text
{
  vision, robot, hardware, updated_at,          # SharedState
  mode: {camera, model, plc},
  frozen_vision | null,
  capture_step: "vision" | "robot",
  capture_feedback | null,
  candidate: null,                               # sempre null nesta versão
  capture_readiness,
  active_session_id | null,
  active_plan_z | null,
  session | null                                 # session_detail se houver sessão ativa
}
```

`capture_readiness`:

```text
{
  session, plan, step, single_instance, confidence, mask_not_cut,
  axis_quality, mask_area, stable, pose, region, plan_z, not_duplicate,
  candidate_clear,   # true quando não há visão congelada
  ready
}
```

`ready` no passo visão: há sessão, plano, próximo índice ≤ 9 e `vision is not None`. No passo robô: as mesmas pré-condições de sessão/plano/índice e `robot.fresh`. Os gates individuais não entram em `ready`.

### 16.4 `GET /api/frame`

JPEG do último overlay. **503** `{"detail": "Frame ainda indisponível"}` se ainda não houve encode. A HMI pede de novo a cada **280 ms** com query `t` para furar cache.

### 16.5 `GET /api/config`

Projeção pública. Não devolve IP do CLP nem caminho do checkpoint.

```text
{
  version, default_planes_mm, xy_tolerance_mm, angular_tolerance_deg,
  pick_offset_local_mm, model_threshold,
  mode: {camera, model, plc},
  model_hash, git_revision, pixel_reference
}
```

### 16.6 Sessões

`GET /api/sessions` → lista decrescente por `created_at`.

`POST /api/sessions` **201**. Corpo:

```text
{ "name": str(1..120), "planes": [float, ...] }
```

Recusas 422: nome vazio, lista vazia, não finito, plano duplicado.

`GET /api/sessions/{session_id}` → `session_detail`.

`POST /api/sessions/{session_id}/activate` → marca `active`, torna-a a sessão corrente, limpa plano e congelamento.

`POST /api/sessions/{session_id}/plans/{plan_z}/activate` → `plan_z` tem de estar em `planes_mm`. Ativa a sessão se ainda não for a corrente. Limpa congelamento.

### 16.7 Captura

`POST /api/capture` sem corpo.

Resposta do 1/2:

```text
{ saved: false, step: "vision_frozen", frozen: FrozenVisionCapture }
```

Resposta do 2/2:

```text
{ saved: true, step: "registered", pair, evaluation, feedback }
```

`POST /api/candidate/decision` corpo `{confirm: bool}`.

- `true` → 422 `O ponto é gravado em Capturar coordenadas do robô (2/2)`;
- `false` sem congelamento → 422;
- `false` com congelamento → `{saved:false, candidate_id, step:"vision"}`.

### 16.8 `POST /api/sessions/{session_id}/export`

```text
{ "json": "/exports/sincro_robo_<8>.json", "csv": "/exports/sincro_robo_<8>_pontos.csv" }
```

### 16.9 Estáticos

| Mount | Diretório |
| --- | --- |
| `/assets` | `sincro_robo/web` |
| `/exports` | `app.exports_dir` |

---

## 17. HMI

Ficheiros: `sincro_robo/web/index.html`, `app.js`, `styles.css`. Idioma `pt-BR`. Sem framework. Estado em memória no objeto `app`.

| Superfície | Recurso mostrado | Fonte |
| --- | --- | --- |
| relógio | hora local, 500 ms | `Date` do browser |
| chips Câmera / RF-DETR / CLP | status do hardware | `/api/state` |
| chip CLP | pisca no eco quando online; eco morto = falha | `hardware.plc` |
| banner de modo | providers e `v{version} · {git}` | `/api/config` |
| frame | JPEG | `/api/frame` |
| leituras C, vetor, VCPn, Q | `native_*` e heading | `state.vision` |
| confiança e qualidade de eixo | barras | `state.vision` |
| σX σY σθ | jitter | `state.vision` |
| pose X Y Z Rx Ry Rz | último snapshot; frescura | `state.robot` |
| gates | 9 indicadores; não bloqueiam | `capture_readiness` |
| botão 1/2 | `CAPTURAR COORDENADAS DA VISÃO (1/2)` | `capture_step=vision` |
| botão 2/2 | `CAPTURAR COORDENADAS DO ROBÔ (2/2)` | `capture_step=robot` e `ready` |
| descartar | visível só com congelamento | `POST decision confirm=false` |
| planos | bolhas de ajuste/validação, alvo, RMSE | `session.plans` |
| métricas | RMS XY, P95 XY, RMS θ, máximo, estado passed | plano ativo |
| tabela | pares da sessão | `session.pairs` |
| diálogo nova sessão | nome + planos separados por vírgula; default `0, 200, 400` | `POST /api/sessions` + activate |

O botão de captura fica desabilitado quando `capture_readiness.ready` é falso. No 2/2 isso coincide com eco morto. No 1/2 coincide com ausência de visão, de sessão ou de plano.

---

## 18. Overlay publicado no JPEG

Contrato do perfil de campo: `annotate_vcpn_overlay` (o loop pede `overlay="vcpn"`). Cores BGR.

| Elemento | Cor BGR | Geometria |
| --- | --- | --- |
| preenchimento da máscara | `(0, 255, 128)` com alfa 0,40 | contornos externos, contorno espessura 3 |
| retângulo ROI | `(0, 255, 0)` | `roi_px` |
| bússola N/L | `(255, 255, 0)` | âncora da aresta superior, braço 18 px |
| centroide C | `(255, 255, 255)` | raio 6 |
| seta C→VCPn | `(0, 255, 255)` | linha + ponta 14×10 |
| ponto VCPn | `(0, 0, 255)` | raio 6 |
| HUD | branco com contorno preto | canto (8, 18), passo 16 px |

Linhas do HUD, nesta ordem: rótulo `Embalagem`; `conf:{inteiro %}`; `C`; `CX:{int}`; `CY:{int}`; `Vetor`; `ang:{int}deg`; `VCPn`; `X:{int}`; `Y:{int}`; `Q:{quadrante}` se houver; `A:{:.1f}cm2` se a área existir.

`annotate_pick_overlay` (máscara, AABB, eixo magenta, arco, seta) existe e é testado, mas o loop de campo não o usa.

---

## 19. Temporização de referência

| Relógio | Período | Onde |
| --- | --- | --- |
| grab + inferência | 125 ms alvo (8 fps), mais o tempo de GPU | `camera-owner` |
| poll CIP | 200 ms | `cip-owner` |
| toggle heartbeat | 1,0 s | escrita BOOL |
| perda de eco | 3,0 s sem aresta | watchdog |
| JPEG no browser | 280 ms | `app.js` |
| estado no browser | 700 ms | `app.js` |
| relógio de UI | 500 ms | `app.js` |
| simulador troca de pose visual | 10 s por região | `SyntheticCamera` |
| join no shutdown | 5 s por thread | `ApplicationRuntime.stop` |
| git rev-parse | timeout 3 s | hash de revisão na arranque |

Não há fila, não há carimbo de sincronismo visão↔CIP no mesmo instante, e não há interpolação temporal. O par gravado junta a visão congelada no 1/2 com a pose lida no instante do 2/2.

---

## 20. Dependências

Obrigatórias (`pyproject.toml`):

| Pacote | Faixa |
| --- | --- |
| fastapi | `>=0.115,<1` |
| uvicorn[standard] | `>=0.34,<1` |
| numpy | `>=2,<3` |
| opencv-python-headless | `>=4.10,<5` |
| Pillow | `>=10,<13` |

Extra `pcbox`:

| Pacote | Faixa |
| --- | --- |
| aphyt | `>=0.1.24,<1` |
| rfdetr | `==1.10.1` |
| stapipy | wheel local 1.2.3 cp312 linux_x86_64 (fora do índice) |
| PyTorch/CUDA | o que o RF-DETR 1.10.1 exigir no PCBOX; não está pinado neste repositório |

Extra `dev`: `httpx>=0.27,<1`, `pytest>=8,<10`.

Preflight (`scripts/preflight.py`) exige, todos `ok`:

- CPython 3.12;
- arquitetura `x86_64` ou `amd64`;
- ficheiro `/opt/sentech/.stprofile`;
- módulos importáveis: `numpy`, `cv2`, `fastapi`, `uvicorn`, `rfdetr`, `aphyt`, `stapipy`;
- `rfdetr` instalado == `model.rfdetr_version`;
- checkpoint existente (publica SHA-256);
- `plc.ip` não vazio;
- com `--check-network`, TCP 44818 aceita conexão.

Código de saída 0 só se todos os itens incluídos estiverem `ok`.

---

## 21. Comandos que reservam recursos

| Comando | O que reserva |
| --- | --- |
| `./run.sh` ou `./run.sh pcbox` | processa `config/pcbox.json`, source do StApi, HTTP 8080, câmera, CIP, escrita do heartbeat |
| `./run.sh simulator` | HTTP em 127.0.0.1, sem hardware; a suíte de testes usa este perfil |
| `python -m sincro_robo --config <json> [--host] [--port]` | igual, com overrides de socket |
| `scripts/install_pcbox.sh` | cria `.venv`, instala `.[pcbox]` e o wheel StApi |
| `python scripts/preflight.py --config config/pcbox.json [--check-network]` | nenhum device; no máximo um TCP connect |
| `python -m pytest` | simulador; não abre StApi nem CIP |

Perfil operacional de campo é o PCBOX. O simulador não é modo de produção.

---

## 22. Matriz de recursos para o especificador

Use esta tabela como lista de verificação. Cada linha é um recurso que precisa de dono, quantidade e exclusividade no documento de campo.

| ID | Recurso | Quantidade | Exclusividade | Especificar |
| --- | --- | --- | --- | --- |
| R1 | PCBOX x86_64, CPython 3.12 | 1 | processo SINCRO | CPU, RAM, disco para `data/` e checkpoint |
| R2 | GPU/CUDA compatível com RF-DETR 1.10.1 | 0 ou 1 | processo SINCRO durante inferência | dispositivo, VRAM, driver |
| R3 | SentechSDK 1.2.3 em `/opt/sentech` | 1 | máquina | versão, `.stprofile`, wheel stapipy |
| R4 | Câmera Omron Sentech, device index 0 | 1 | um cliente StApi | modelo, formato de pixel, IP/USB, 2592×1944 |
| R5 | Omron NX102 | 1 | rede de célula | IP, máscara, gateway |
| R6 | Porta EtherNet/IP 44818 | 1 conexão explícita | sessão aphyt na thread CIP | timeout 10 s |
| R7 | Tag `RobFrom_Coord_CurrBase_Tool[0..5]` | 6 valores | leitura | tipo Sysmac, unidade linear, unidade angular |
| R8 | Tag `VisionCtrl_Heartbeat` | 1 BOOL | um escritor | programa que faz o eco; Realtec Vision parado |
| R9 | Tag `PlcCtrl_HeartBeat` | 1 BOOL | leitura | lógica de espelho e tempo de resposta &lt; 3 s |
| R10 | TCP 8080 no PCBOX | 1 listen | processo SINCRO | interface (`0.0.0.0`), clientes HMI |
| R11 | Bundle `checkpoint_best_total.pth` | 1 ficheiro | leitura | caminho, SHA-256, classe `Molde`, threshold 0,3 |
| R12 | SQLite `data/sincro_robo.sqlite3` | 1 | processo SINCRO | backup, retenção |
| R13 | Diretório `data/exports` | N ficheiros | escrita local | retenção de JSON/CSV |
| R14 | Offset visual 55 mm, 1,026 mm/px | constantes de plano Z=200 | — | confirmar por plano 0 e 400 |
| R15 | Offset de garra `[0, 55]` mm no frame da ferramenta | 1 vetor | — | eixo local (Y) e sinal |
| R16 | Planos Z 0 / 200 / 400 mm | 3 por sessão default | — | lista real da campanha |
| R17 | 7 pontos (5 ajuste + 2 validação), teto 9 | por plano | — | regiões físicas correspondentes |
| R18 | Tolerâncias 20 mm e 5° | limites provisórios | — | valores finais; o código não os reduz sozinho |
| R19 | Usuário `sincro-robo` e `/etc/sincro-robo.env` | 1 | unit systemd | só depois da validação de exclusão da câmera |
| R20 | Repositório e tag Git | 1 revisão | implantação | owner GitHub; não copiar árvore solta para o PCBOX |

---

## 23. Itens em aberto para especificação de campo

O código assume os valores abaixo e **não os descobre** no CLP nem na câmera. Fechar cada um antes de tratar a calibração como medida.

1. Tipo Sysmac do array `RobFrom_Coord_CurrBase_Tool` (REAL ou LREAL) e unidades efetivas de X/Y/Z e de Rx/Ry/Rz.
2. IP do NX102 se não for `192.168.250.1`.
3. Convenção do heading visual contra o molde físico (0° leste, 90° norte de imagem, vetor sem componente sul).
4. Sinal e eixo de `pick_offset_local_mm = [0, 55]`: Y local da ferramenta, distinto do offset de imagem C→VCPn.
5. `mm_per_px = 1,026` vale para o plano Z = 200 mm no frame 960×720. Planos 0 e 400 mm não têm escala própria.
6. Tolerâncias finais. 20 mm e 5° são tetos provisórios da POC; a expansão para 9 pontos não altera esses números.
7. Owner e URL do repositório GitHub usados na implantação por tag.
8. Se `plc.freshness_timeout_s` deve passar a significar alguma coisa; hoje é configuração morta.
9. Se `model.device` deve ser aplicado ao `RFDETRSegSmall`; hoje `"auto"` é ignorado.
10. Programa do NX102 que espelha `VisionCtrl_Heartbeat` em `PlcCtrl_HeartBeat`, incluindo o atraso máximo aceitável dentro da janela de 3 s.

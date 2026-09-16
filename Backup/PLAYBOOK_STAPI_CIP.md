# Playbook StApiPy + CIP — referência para outra aplicação

Cookbook **sem Qt** para reutilizar o **mesmo hardware** e os **mesmos métodos de conexão** deste repositório (branch StApi). Destinado a uma **ferramenta à parte** (diagnóstico/calibração): só liga com o supervisório Buddemeyer **parado**.

Este software **não** é uma dependência da nova aplicação. As classes citadas são implementação de referência.

| Classe / ficheiro | Papel neste repo |
|-------------------|------------------|
| `realtec_vision_buddmeyer/streaming/stapipy_adapter.py` | Câmera Omron Sentech via `stapipy` |
| `realtec_vision_buddmeyer/communication/cip_client.py` | CIP EtherNet/IP via `aphyt` (`NSeries`) |
| `realtec_vision_buddmeyer/communication/tag_map.py` | Padrão mapa lógico → nome físico + whitelist |
| `Iniciar_Realtec_Vision.sh` | `source /opt/sentech/.stprofile` antes do Python |

---

## 0. Contrato deste documento

### 0.1 O que está fechado

| Tema | Acordo |
|------|--------|
| Célula (câmera + CIP) | Lubuntu 26.04 **x86_64**, **CPython 3.12** |
| Câmera | SentechSDK **1.2.3** + wheel `stapipy` **cp312** `linux_x86_64` + `/opt/sentech/.stprofile` |
| Câmera **não** usa | Harvester, ficheiro `.cti`, GenTL como API de aplicação |
| macOS | Laboratório: CIP e/ou fonte de imagem local. **StApi GigE não está comprovado neste repo** |
| Papel da nova app | Ferramenta à parte; não convive com este supervisório nem com o StViewer no mesmo device |
| CIP | `aphyt`, EtherNet/IP, ler e escrever; **sem PySide6** |
| Tags | **Não copiar** o handshake `Vision*` / robô deste supervisório. Preencher o mapa na implementação da nova app |

### 0.2 Legenda de evidência

- **Fato (código):** observado neste repositório.
- **Neste software (config):** valor no `config.yaml` / defaults Pydantic; **confirmar na célula**.
- **Não verificado:** não inventar; preencher com medição, export Sysmac ou manual Omron.

### 0.3 Fora de escopo

- UI PySide6, FSM pick-and-place, visão/Mask2Former, calibração px→mm.
- Contrato de tags deste supervisório (`TAG_CONTRACT.md`, `write_detection_result`, `VisionReady`, …).
- Implantar, forçar I/O, contornar permissivos/safety, ou escrever no NX102 sem mapa de tags da **nova** app e autorização local.

---

## 1. Exclusão de hardware (obrigatório)

A câmera StApi é **um cliente por device**. Este software recusa abrir se outro processo (StViewer ou o supervisório) ainda segura o handle.

Antes de ligar a ferramenta:

1. Parar o supervisório Buddemeyer (Play/Stop e processo Python).
2. Fechar **StViewer** e qualquer outro cliente GigE/StApi.
3. Confirmar que nenhum serviço systemd deste repo ficou a correr (`realtec-vision`, se existir).
4. Só então iniciar a nova app.

Dois clientes CIP no mesmo NX102 **podem** coexistir ao nível EtherNet/IP; mesmo assim **não** escreva nas tags deste supervisório. A nova app usa **outro conjunto de tags**, a informar na implementação.

Escrita no CLP real: autorização, backup Sysmac, janela e rollback. Este playbook não substitui safety/LOTO.

---

## 2. Plataformas

| Plataforma | Câmera StApi | CIP (`aphyt`) | Python |
|------------|--------------|---------------|--------|
| Lubuntu 26.04 x86_64 | Sim — mesmo método deste software | Sim | **3.12** (ABI do wheel) |
| macOS (lab) | Não documentado aqui | Sim | 3.10+ |

**Não verificado:** o SentechSDK 1.2.3 ser oficialmente suportado no Lubuntu/Ubuntu **26.04**. Neste repo o launcher Linux e o wheel `linux_x86_64` são o caminho de produção; se o instalador Omron recusar o SO, isso é bloqueio de fabricante, não de Python.

**Não verificado neste repo:** wheel `stapipy` para macOS (Intel ou Apple Silicon).

---

## 3. Câmera — Omron Sentech via StApiPy (Lubuntu)

### 3.1 Stack (fato)

```
Câmera GigE (Omron Sentech)
    → SentechSDK 1.2.3 (driver + runtime)
    → /opt/sentech/.stprofile  (LD_LIBRARY_PATH e afins)
    → import stapipy   (wheel Omron, NÃO PyPI)
    → initialize → create_system → device → datastream → acquire
```

Não há ficheiro `.cti` na produção actual. `gentl_cti_path` no YAML é legado e ignorado.

Índice de device neste software: `streaming.gentl_device_index` (default **0** = `create_first_device()`).

Modelo de câmara: **não congelar neste playbook**. Docs antigos de GenTL falam em STC-MCS2041POE; testes usam um `display_name` fake. Na célula, use o primeiro device que o StApi enumerar, ou confirme o modelo no StViewer **com a ferramenta parada**.

### 3.2 Instalação do runtime (Lubuntu)

1. Instalar **SentechSDK 1.2.3** da Omron (instalador Linux do fabricante). Caminho esperado do perfil: `/opt/sentech/.stprofile`.
2. Criar venv **só** com CPython 3.12 x86_64:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-python numpy
# Wheel NÃO está no PyPI — use o ficheiro do SDK / pasta Omron:
pip install /caminho/stapipy-1.2.3-cp312-cp312-linux_x86_64.whl
```

3. **Toda** invocação Python da câmera deve ter o perfil carregado. Neste repo o launcher faz:

```bash
if [ -f /opt/sentech/.stprofile ]; then
    set +u
    source /opt/sentech/.stprofile
    set -u
fi
```

`set +u` é necessário: o `.stprofile` da Omron concatena `LD_LIBRARY_PATH` mesmo quando a variável ainda não existe (`set -u` rebenta o `source`).

4. Confirmar:

```bash
source /opt/sentech/.stprofile   # com set +u se usar nounset
python -c "import stapipy as st; print(st)"
```

### 3.3 Regras de thread (crítico)

Fato neste software (`StapipyAdapter` + `StreamManager`):

- `st.initialize()`, `create_system`, device, datastream, `retrieve_buffer`, `acquisition_stop`, `stop_acquisition` e `st.terminate()` correm **na mesma thread** (a de captura).
- Chamar `acquisition_stop` / `terminate` na thread da UI **aborta o GenICam**.
- `close()` noutro thread é recusado (`stapipy_close_wrong_thread`).

Na ferramenta nova: uma thread (ou o thread principal, se for CLI) é dona de **todo** o ciclo StApi.

### 3.4 Ciclo de vida (copiar)

Ordem de **open** (fato: `StapipyAdapter.open`):

1. `st.initialize()`
2. `system = st.create_system()`
3. Device:
   - índice `<= 0`: `system.create_first_device()`
   - índice `> 0`: `system.create_first_interface()` + `create_device_by_index(index)` — se a API não expor isto, este software falha e pede índice 0
4. `datastream = device.create_datastream()`
5. `datastream.start_acquisition()`
6. `device.acquisition_start()`

Ordem de **close** (fato: `_teardown`):

1. `device.acquisition_stop()`
2. `datastream.stop_acquisition()`
3. largar referências
4. `st.terminate()` se `initialize` chegou a correr

### 3.5 Captura de frame

```python
retrieved = datastream.retrieve_buffer(timeout_ms)
```

- `timeout_ms` neste software: **400** ms (`fetch_timeout_ms`).
- **Não** passar um segundo argumento de “timeout handling”. Comentário no código: o sample Omron usa `retrieve_buffer()` sem `EStTimeoutHandling`; um 2.º argumento inválido aborta em GenICam C++ (`LogicalErrorException` / `std::terminate`).
- Se o objecto for context manager: `with retrieved as buffer:`.
- Caso contrário: `get_image()` e `release()` se existir.
- Copiar pixels **antes** de sair do `retrieve` (`np.frombuffer(...).copy()`), como `grab_opencv.py` / `_image_to_bgr`.
- Ignorar buffer sem imagem: `buffer.info.is_image_present is False`.

### 3.6 Bayer → BGR (OpenCV)

Fato: GenICam/StApi `BayerRG` = RGGB. Os aliases **de duas letras** do OpenCV (`COLOR_BayerRG2BGR`) correspondem a **BGGR** (troca R/B). Este software usa aliases de **quatro letras**:

| Token GenICam / filtro | OpenCV |
|------------------------|--------|
| BayerRG | `cv2.COLOR_BayerRGGB2BGR` |
| BayerGR | `cv2.COLOR_BayerGRBG2BGR` |
| BayerGB | `cv2.COLOR_BayerGBRG2BGR` |
| BayerBG | `cv2.COLOR_BayerBGGR2BGR` |

Fallback se o filtro for desconhecido: `cv2.COLOR_BayerGRBG2BGR`.

Mono → `cv2.COLOR_GRAY2BGR`. Bits > 8: `uint16` escalado para `uint8` com `each_component_valid_bit_count` quando `get_pixel_format_info` existir.

Resize é política **desta HMI** (teto típico 960 px, segurança 1920). A ferramenta nova **não precisa** redimensionar para obter um frame válido.

### 3.7 Nós GenICam (opcional)

Com o device aberto, nodemap: `device.remote_port.nodemap`.

Nós lidos/escritos neste software, **se existirem** no device:

- `Gain`, `ExposureTime` ou `ExposureTimeAbs`
- `ExposureAuto`, `GainAuto`

Acesso: `nodemap.get_node(nome)` ou atributo; escrita `node.value = ...`.

### 3.8 Snippet mínimo (CLI, uma thread)

Pré-requisito: `source /opt/sentech/.stprofile` no mesmo shell.

```python
"""Grab único — Lubuntu, CPython 3.12, SentechSDK 1.2.3 + stapipy cp312."""
from __future__ import annotations

import threading

import cv2
import numpy as np
import stapipy as st

FETCH_TIMEOUT_MS = 400
BAYER_TO_BGR = {
    "BayerRG": cv2.COLOR_BayerRGGB2BGR,
    "BayerGR": cv2.COLOR_BayerGRBG2BGR,
    "BayerGB": cv2.COLOR_BayerGBRG2BGR,
    "BayerBG": cv2.COLOR_BayerBGGR2BGR,
}


def _bayer_code(pixel_format, info) -> int:
    filt = None
    if info is not None and hasattr(info, "get_pixel_color_filter"):
        try:
            filt = info.get_pixel_color_filter()
        except Exception:
            filt = None
    enum = getattr(st, "EStPixelColorFilter", None)
    if filt is not None and enum is not None:
        mapping = (
            (getattr(enum, "BayerRG", None), BAYER_TO_BGR["BayerRG"]),
            (getattr(enum, "BayerGR", None), BAYER_TO_BGR["BayerGR"]),
            (getattr(enum, "BayerGB", None), BAYER_TO_BGR["BayerGB"]),
            (getattr(enum, "BayerBG", None), BAYER_TO_BGR["BayerBG"]),
        )
        for key, code in mapping:
            if key is not None and filt == key:
                return code
    text = str(filt or pixel_format or "").lower()
    for token, code in BAYER_TO_BGR.items():
        if token.lower() in text:
            return code
    return cv2.COLOR_BayerGRBG2BGR


def buffer_to_bgr(st_image) -> np.ndarray:
    data = st_image.get_image_data()
    width, height = int(st_image.width), int(st_image.height)
    pixel_format = getattr(st_image, "pixel_format", "")
    info = None
    try:
        info = st.get_pixel_format_info(pixel_format)
    except Exception:
        info = None
    bits = int(getattr(info, "each_component_total_bit_count", 8) or 8) if info else 8
    is_bayer = "bayer" in str(pixel_format).lower()
    is_mono = "mono" in str(pixel_format).lower()
    if info is not None:
        is_bayer = bool(getattr(info, "is_bayer", is_bayer))
        is_mono = bool(getattr(info, "is_mono", is_mono))
    if bits > 8:
        nparr = np.frombuffer(data, np.uint16).copy()
        valid = int(getattr(info, "each_component_valid_bit_count", bits) or bits) if info else bits
        div = pow(2, max(0, valid - 8))
        nparr = (nparr / div).astype(np.uint8) if div else nparr.astype(np.uint8)
    else:
        nparr = np.frombuffer(data, np.uint8).copy()
    nparr = nparr.reshape(height, width, 1)
    if is_bayer:
        nparr = cv2.cvtColor(nparr, _bayer_code(pixel_format, info))
    elif is_mono or nparr.shape[2] == 1:
        nparr = cv2.cvtColor(nparr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(nparr)


def main() -> None:
    owner = threading.get_ident()
    initialized = False
    device = None
    datastream = None
    try:
        st.initialize()
        initialized = True
        system = st.create_system()
        device = system.create_first_device()
        datastream = device.create_datastream()
        datastream.start_acquisition()
        device.acquisition_start()
        retrieved = datastream.retrieve_buffer(FETCH_TIMEOUT_MS)
        if retrieved is None:
            raise RuntimeError("timeout retrieve_buffer")
        if hasattr(retrieved, "__enter__"):
            with retrieved as buffer:
                image = buffer_to_bgr(buffer.get_image())
        else:
            image = buffer_to_bgr(retrieved.get_image())
            release = getattr(retrieved, "release", None)
            if callable(release):
                release()
        print("frame", image.shape, "thread", threading.get_ident() == owner)
        cv2.imwrite("stapi_grab.png", image)
    finally:
        if threading.get_ident() != owner:
            raise RuntimeError("teardown noutro thread — abortar")
        if device is not None:
            try:
                device.acquisition_stop()
            except Exception:
                pass
        if datastream is not None:
            try:
                datastream.stop_acquisition()
            except Exception:
                pass
        if initialized:
            st.terminate()


if __name__ == "__main__":
    main()
```

Critério de sucesso: ficheiro `stapi_grab.png` com geometria nativa (ou próxima) e cores alinhadas ao StViewer (R/B não invertidos).

### 3.9 Falhas típicas da câmera

| Sintoma | Causa habitual (código / operação) |
|---------|--------------------------------------|
| `import stapipy` falha | Wheel em falta, Python ≠ 3.12, ABI errado |
| `initialize` / open falha | `.stprofile` não sourced; SDK 1.2.3 em falta |
| Open: outro cliente segura o device | Supervisório ou StViewer ainda a correr |
| Processo morre em `retrieve_buffer` | 2.º argumento inválido no retrieve |
| Cores invertidas | Alias OpenCV de duas letras |
| Close trava / aborta | `terminate` noutro thread |

IP GigE da câmara, jumbo frames, NIC dedicada: **não verificados neste playbook**. Configurar com as ferramentas do SentechSDK (StViewer) com a app Python **parada**.

---

## 4. CIP — Omron NX102 via aphyt (Lubuntu e macOS)

### 4.1 Stack (fato)

```
PC Python
    → pip install aphyt>=0.1.24
    → from aphyt import omron
    → plc = omron.n_series.NSeries()
    → plc.connect_explicit(ip, connection_timeout=...)
    → plc.read_variable(nome_fisico) / plc.write_variable(nome_fisico, valor)
    → plc.close_explicit() ou close()
```

- Biblioteca **Python pura** (funciona em Lubuntu e macOS).
- Controlador alvo neste software: **Omron NX102**.
- Porta EtherNet/IP **44818** está em `cip.port` (default e `config.yaml`).
- **Fato de implementação:** `_connect_sync` chama `connect_explicit(ip, connection_timeout=timeout)` **sem passar a porta**. A sessão usa o default da `aphyt`/`NSeries` (EtherNet/IP standard). Mantenha o NX102 a publicar CIP nessa porta; se a célula usar outra porta, **não verificado** neste repo — validar na `aphyt` da versão instalada.

### 4.2 Parâmetros neste software (confirmar na célula)

Valores em `realtec_vision_buddmeyer/config/config.yaml` (secção `cip`), **não** tratados como verdade de campo sem confirmação:

| Chave | Valor neste YAML | Default Pydantic (`CIPSettings`) |
|-------|------------------|----------------------------------|
| `ip` | `192.168.250.1` | `192.168.0.10` |
| `port` | `44818` | `44818` |
| `connection_timeout` | `10.0` s | `10.0` |
| `io_retries` | `2` (até 3 tentativas por I/O) | `2` |
| `auto_reconnect` | `true` | `true` |

A nova app deve ter o **IP do NX102 da célula** em configuração (`PLC_IP`). Não copiar um IP “porque está no git” sem ping/Sysmac.

Reachability neste repo (preflight, só com `production_mode`): TCP `create_connection((ip, port), timeout=3)`.

### 4.3 Sessão, thread e lock asyncio

Fato: `CIPClient` usa **um** `ThreadPoolExecutor(max_workers=1)` para todo o I/O aphyt. Reutilizar o handle noutra thread/loop dispara erros de `asyncio.Lock` / “sessão CIP anterior não foi libertada”.

Na ferramenta sem Qt:

- Uma thread (ou o main thread) é dona de `NSeries`.
- `connect` → I/O → `close_explicit` nessa mesma dona.
- Antes de voltar a `connect`, fechar a sessão anterior (`close_explicit`). Não reutilizar um objecto “a meio”.
- Não partilhar a instância entre processos.

`read_variable` / `write_variable` usam o **nome físico no CLP** (publicado em EtherNet/IP), não o nome lógico Python.

### 4.4 Tipos e whitelist (padrão, não as tags deste repo)

Copiar o **padrão**, não as entradas de `TagMap.DEFINITIONS`:

```text
logical_name  →  plc_name (Sysmac / EtherNet/IP)
tag_type      →  bool | int | real | string
direction     →  read | write | both
```

Validar tipo **antes** de escrever. Recusar nomes fora da whitelist da **nova** app.

`TAG_CONTRACT.md` deste repo é o contrato do **supervisório**. Nomes físicos lá são proposta de implementação; o próprio documento diz que **não estão verificados** contra o export Sysmac actual. **Não os use** na ferramenta nova.

### 4.5 I/O (fato)

```python
value = plc.read_variable("NomeFisicoNoCLP")
plc.write_variable("NomeFisicoNoCLP", value)
```

Retries neste software: `0.2` s entre tentativas; `io_retries` extra após a primeira.

Bug conhecido da `aphyt`: `RecursionError` em **certos** tags. Este cliente trata isso como falha de biblioteca (valor default seguro ou erro). Na nova app: logar o nome físico e **não** fingir dado fresco.

Queda de sessão (marcadores usados aqui): `ConnectionError`, `TimeoutError`, reset TCP, “event loop” / lock asyncio, download Sysmac. Depois disso: não escrever; `close` e nova `connect`.

Validação de sessão **deste** supervisório escreve `VisionHeartbeat` e lê `RobotReady` — **não reproduzir**. Para a ferramenta: uma leitura/escrita de um tag **vosso**, já publicado e autorizado, ou só `connect_explicit` + uma leitura inócua acordada com o programador do NX102.

### 4.6 Snippet mínimo (sem Qt)

Preencha `PLC_IP` e o mapa. **Não execute escrita** até o mapa da nova app estar fechado com o Sysmac.

```python
"""Sessão CIP síncrona — mesmo transporte que CIPClient._connect_sync."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aphyt import omron

PLC_IP = "PREENCHER"  # neste repo aparece 192.168.250.1 no YAML; confirmar na célula
CONNECT_TIMEOUT_S = 10.0


class TagType(str, Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "real"
    STRING = "string"


class Direction(str, Enum):
    READ = "read"
    WRITE = "write"
    BOTH = "both"


@dataclass(frozen=True)
class Tag:
    logical: str
    plc_name: str
    tag_type: TagType
    direction: Direction


# Preencher na implementação da nova aplicação. Deixar vazio = não há I/O.
TAGS: dict[str, Tag] = {
    # "exemplo_livre": Tag("exemplo_livre", "NomePublicadoNoNX102", TagType.BOOL, Direction.BOTH),
}


def _check_type(tag: Tag, value: Any) -> None:
    if tag.tag_type == TagType.BOOL and not isinstance(value, bool):
        raise TypeError(tag.logical)
    if tag.tag_type == TagType.INT and not isinstance(value, int):
        raise TypeError(tag.logical)
    if tag.tag_type == TagType.REAL and not isinstance(value, (int, float)):
        raise TypeError(tag.logical)
    if tag.tag_type == TagType.STRING and not isinstance(value, str):
        raise TypeError(tag.logical)


class CipSession:
    def __init__(self, ip: str, timeout_s: float = CONNECT_TIMEOUT_S) -> None:
        self._ip = ip
        self._timeout_s = timeout_s
        self._plc: Any = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cip")

    def connect(self) -> None:
        self.close()

        def _open():
            plc = omron.n_series.NSeries()
            plc.connect_explicit(self._ip, connection_timeout=self._timeout_s)
            return plc

        self._plc = self._pool.submit(_open).result()

    def close(self) -> None:
        plc = self._plc
        self._plc = None
        if plc is None:
            return

        def _close():
            method = getattr(plc, "close_explicit", None) or getattr(plc, "close", None)
            if callable(method):
                method()

        try:
            self._pool.submit(_close).result(timeout=5)
        except Exception:
            pass

    def read(self, logical: str) -> Any:
        tag = TAGS[logical]
        if tag.direction not in (Direction.READ, Direction.BOTH):
            raise PermissionError(logical)
        if self._plc is None:
            raise RuntimeError("não conectado")
        return self._pool.submit(self._plc.read_variable, tag.plc_name).result()

    def write(self, logical: str, value: Any) -> None:
        tag = TAGS[logical]
        if tag.direction not in (Direction.WRITE, Direction.BOTH):
            raise PermissionError(logical)
        _check_type(tag, value)
        if self._plc is None:
            raise RuntimeError("não conectado")
        self._pool.submit(self._plc.write_variable, tag.plc_name, value).result()

    def shutdown(self) -> None:
        self.close()
        self._pool.shutdown(wait=True)


if __name__ == "__main__":
    if PLC_IP == "PREENCHER" or not TAGS:
        raise SystemExit("Preencha PLC_IP e TAGS antes de I/O real.")
    session = CipSession(PLC_IP)
    try:
        session.connect()
        # Exemplo: session.read("exemplo_livre")
    finally:
        session.shutdown()
```

Dependência: `pip install "aphyt>=0.1.24"` (mesmo piso que `requirements.txt` deste repo).

### 4.7 Falhas típicas CIP

| Sintoma | Causa habitual |
|---------|----------------|
| `aphyt` ImportError | `pip install aphyt` no venv |
| Timeout de connect | IP errado, VLAN, NX102 off, firewall, porta 44818 fechada |
| Tag error | Nome não publicado no EtherNet/IP; tipo errado; programa Sysmac diferente |
| “sessão anterior não libertada” | `connect` sem `close_explicit`; lock asyncio noutro loop |
| RecursionError | Bug aphyt em tag específico |
| Escrita sem efeito no robô | Tag errado **ou** lógica Sysmac a ignorar; **não** copiar o handshake deste supervisório |

---

## 5. Mapa de tags da nova aplicação (a preencher)

Não há nomes neste playbook. Na implementação, fechar com o programador do NX102:

| Lógico | Nome físico Sysmac | Tipo CIP | Direcção | Unidade / semântica | Publicado EtherNet/IP? |
|--------|-------------------|----------|----------|---------------------|------------------------|
| *(vazio)* | | bool / int / real / string | read / write / both | | |

Critério: o nome físico tem de existir no export/publicação EtherNet/IP **actual**, não neste git.

---

## 6. Arranque recomendado da ferramenta

### Lubuntu (câmera + CIP)

```bash
# 1. Supervisório e StViewer parados
# 2. Runtime Sentech
set +u
source /opt/sentech/.stprofile
set -u
source .venv/bin/activate   # python 3.12
python grab_e_ou_cip.py
```

### macOS (só CIP / lab)

```bash
source .venv/bin/activate   # 3.10+
pip install "aphyt>=0.1.24"
# Sem stapipy. Sem GigE Sentech neste playbook.
python cip_lab.py
```

Rede: PC e NX102 na mesma sub-rede EtherNet/IP. Câmera GigE: NIC/rede conforme o SDK Omron (**não verificado** neste texto).

---

## 7. Teste de confirmação / rollback

| Passo | Critério de sucesso | Risco | Rollback |
|-------|---------------------|-------|----------|
| Import `stapipy` no 3.12 com `.stprofile` | Módulo carrega | Baixo | Desactivar venv |
| Grab com supervisório parado | PNG válido; device libertado no `terminate` | Ocupar a câmara | Fechar processo; `terminate` na thread dona |
| `connect_explicit` ao NX102 | Sessão abre; `close_explicit` devolve | Tráfego CIP | Fechar sessão; não deixar handle órfão |
| Read/write nas **vossas** tags | Valor lido/escrito bate com watch Sysmac | Movimento / lógica de célula se o tag for de comando | Só tags acordadas; backup do programa NX102 **antes** |

Não use os testes pytest deste repo (`InMemoryPLC`) como prova de hardware. Esses testes **não** abrem StApi nem o NX102 real.

---

## 8. Referência rápida de versões (fato neste branch)

| Item | Valor no código / requirements |
|------|--------------------------------|
| SentechSDK | 1.2.3 |
| Wheel | `stapipy-1.2.3-cp312-cp312-linux_x86_64.whl` (local) |
| Perfil | `/opt/sentech/.stprofile` |
| `aphyt` | `>=0.1.24` |
| Classe PLC | `aphyt.omron.n_series.NSeries` |
| Porta documentada | 44818 |
| Timeout retrieve StApi | 400 ms |
| Device default | índice 0 |

---

## 9. Manutenção

Quando o adaptador StApi ou o `_connect_sync` deste repo mudarem, actualizar este playbook no mesmo PR. Não acrescente tags, IPs ou modelos de câmara sem fonte (código, Sysmac, ou teste de célula datado).

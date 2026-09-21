# SINCRO_ROBO

Aplicação web local para calibração manual do relacionamento entre a câmera Omron Sentech, a segmentação RF-DETR e a pose de um robô lida no Omron NX102 via CIP.

A implementação segue o **PRD + SDD v0.4** e o `Backup/PLAYBOOK_STAPI_CIP.md`. Ela não comanda movimentos, não executa handshake, não verifica permissivos de segurança e não escreve no CLP.

## O que está implementado

- câmera simulada para desenvolvimento no Mac;
- adaptador StApiPy com todo o ciclo de vida na mesma thread;
- RF-DETR `RFDETRSegSmall` com o checkpoint fornecido;
- centroide por máscara e eixo principal por PCA;
- ângulo horário em coordenadas de imagem, normalizado em `[0°, 180°)`;
- indicador de qualidade do eixo, máscara cortada, confiança, área e jitter;
- leitura CIP read-only de `RobFrom_Coord_CurrBase_Tool[0..5]`;
- tela em tempo real para X/Y/Z/Rx/Ry/Rz;
- sessões com planos Z configuráveis e defaults 0/200/400 mm;
- 5 pontos de ajuste + 2 de validação por plano;
- expansão adaptativa até 9 pontos quando os limites provisórios falham;
- sugestão e gate da próxima região útil do FOV;
- congelamento de visão + pose e confirmação humana obrigatória;
- correção explícita do offset local `(+57,5 mm, 0 mm)` com rotação por Rz;
- transformação afim por plano Z e offset angular módulo 180°;
- métricas XY média/RMS/máximo/P95 e angulares média absoluta/RMS/máximo;
- marcação `suspect` acima de 20 mm ou 5°;
- SQLite, trilha de auditoria e exportações JSON/CSV com versão e hash SHA-256 do modelo.

## Arquitetura

```text
CameraService (thread proprietária)
  ├── SyntheticCamera ou StApiCamera
  └── SyntheticSegmenter ou RFDetrSegmenter
          ↓
    MoldPoseEstimator + VisionStabilityTracker
          ↓
       SharedState ← PLCService (thread CIP proprietária, read-only)
          ↓
    CaptureController ── confirmação humana
          ↓
    SQLite + CalibrationEngine + Export/Audit
          ↓
        FastAPI + HMI web
```

## Executar localmente no Mac

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m sincro_robo --config config/simulator.json
```

Abra [http://127.0.0.1:8080](http://127.0.0.1:8080). O banner deve informar que câmera, modelo e CLP estão simulados. Nenhum hardware é acionado nesse perfil.

Fluxo recomendado:

1. crie a sessão e confirme os planos Z;
2. clique em um plano;
3. espere todos os gates ficarem verdes;
4. clique em **CAPTURAR PONTO**;
5. valide os valores congelados;
6. use **CANCELAR** para descartar ou **GRAVAR** para persistir;
7. acompanhe a região sugerida, progresso e métricas;
8. exporte JSON e CSV ao final.

## Testes

```bash
python -m pytest
```

A suíte cobre normalização em 0°/90°/179°/180°, equivalência axial, PCA, máscara cortada, offset rotativo, ajuste afim, métricas, degeneração geométrica, confirmação obrigatória, cancelamento, duplicidade, gates e API.

## Configuração

- `config/simulator.json`: desenvolvimento local completo.
- `config/pcbox.json`: StApi + RF-DETR + CIP real.
- variável `SINCRO_CONFIG`: escolhe um JSON sem usar `--config`.
- variável `SINCRO_PLC_IP`: sobrescreve o IP do CLP; obrigatória no perfil PCBOX.

Os limites de jitter ficam em configuração porque o PRD determina sua caracterização durante o POC. Os defaults locais são conservadores e não representam tolerância final de produção.

## Bundle RF-DETR

O app lê `manifest.json` como referência operacional e carrega:

```text
NEW_buddmeyer_rfdetr_seg__seg_small__20260915_183353buddmeyer_rfdetr_seg__seg_small__20260915_183353/checkpoint_best_total.pth
```

Parâmetros fixados pelo bundle: `RFDETRSegSmall`, resolução 384, classe `Molde` e threshold inicial 0,3. O checkpoint registra `rfdetr_version = 1.10.1`; a dependência do perfil PCBOX é fixada nessa mesma versão. O bundle `buddmeyer_rfdetr_seg__seg_small__20260915_183353` permanece no repositório só para rollback.

## PCBOX e integração de campo

Veja [DEPLOY_PCBOX.md](DEPLOY_PCBOX.md). A implantação deve ocorrer pelo fluxo Git/GitHub/tag; não copie mudanças sem versionamento diretamente para o PCBOX.

Antes da v1.0 ainda precisam ser confirmados em campo:

- tipo Sysmac exato do array (REAL/LREAL) e unidades efetivas;
- convenção angular visual contra o molde físico;
- sentido `+Xm` do offset de 57,5 mm;
- IP real do NX102;
- tolerâncias finais após a campanha;
- owner/repositório do GitHub.

## Segurança operacional

- A câmera StApi aceita um cliente por device: pare o supervisório e feche o StViewer.
- `StApiCamera` abre, captura e encerra na mesma thread.
- `CipPoseReader` não expõe operação de escrita.
- Variação da pose é apenas diagnóstico; nunca é tratada como `robot_stopped`.
- A persistência de um `RobotPoseSnapshot` só ocorre depois da resposta explícita **GRAVAR**.

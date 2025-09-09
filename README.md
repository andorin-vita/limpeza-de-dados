## Limpeza de Dados

Este projeto tem como objetivo facilitar a limpeza, organização e visualização de dados recolhidos através de formulários Google Forms no âmbito do projecto Andorin

### Funcionalidades
- Processamento automático de dados provenientes de formulários Google.
- Normalização e correção de nomes de localidades (distritos, concelhos, freguesias).
- Geração de ferramentas de dados para uma limpeza rápida.
- Criação de mapas e clusters para visualização geográfica dos dados.

### Estrutura do Projeto
- `src/limpeza_de_dados/`: Módulos principais para limpeza, criação de mapas e utilitários.
- `config/`: Configurações específicas do projeto.
- `dados/`: Dados de entrada e saída.
- `cleaning_tool_app.py`: Streamlit app para validação de novas submissões
- `create_maps_clusters.py`: Streamlit app para visualização dos dados validados usando mapas de clusters
- `create_maps.py`: Streamlit app para visualização dos dados validados usando mapas de pontos

### Como instalar
1. Instalar Python 3.12
2. Instalar as dependências de `requirements.txt` (mais informações sobre como instalar dependências abaixo)
3. Copiar o ficheiro da chave de service account (.json) para a pasta `config` (mais informações sobre como obter uma chave de service account abaixo)
4. Copiar ficheiro `.env-example` e mudar-lhe o nome `.env` (estas são as variáveis de ambiente/segredos)
    - Colocar o nome do ficheiro da service account (.json) na variável `SERVICE_ACCOUNT_PATH`
    - Colocar os urls das Google Sheets nas respetivas variáveis 

### Como executar localmente
1. No ficheiro `scr/create_data_flow_to_google.py`, editar `RUN_IN_STREAMLIT: bool = False` para `RUN_IN_STREAMLIT: bool = True`
2. Num terminal, executar `streamlit run cleaning_tool_app.py` para visualizar no browser a ferramenta de validação de novas submissões.
2. Num terminal, executar `streamlit run create_maps.py` para visualizar no browser a ferramenta de visualização de mapas de pontos.
2. Num terminal, executar `streamlit run create_maps_clusters.py` para visualizar no browser a ferramenta de visualização de mapas de clusters.

### Extra: Como instalar dependências

#### Usando requirements.txt com ambiente virtual

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### Instalação do uv (Gestor de pacotes rápido para Python)
Windows:
```powershell
# Usando Windows PowerShell (com privilégios de administrador)
iwr https://astral.sh/uv/install.ps1 -useb | iex

# OU usando pip
pip install uv
```

#### Usando pyproject.toml com uv

Windows:
```powershell
uv venv
.venv\Scripts\activate
uv sync
```

### Extra: Como obter uma chave de service account

Cada membro da equipa deve ter a sua chave.
Assumindo que já existe um projeto no Google Cloud Console e que os scopes necessários já estão ativados:

1. Fazer o login no Google com `registosandorin@gmail.com`
2. Aceder ao [Google Cloud Console](https://console.cloud.google.com/welcome?project=formandorin)
3. Pesquisar na barra de pesquisa "Service Accounts"
4. Clicar nas Actions/Ações (última coluna da tabela) 
5. Escolher "Gerenciar chaves"/"Manage Keys"
6. Seleccionar "Adicionar chave"/"Add Key"
7. Escolher opção "JSON" no pop-up - isto inicia download de um ficheiro - este é o ficheiro que devemos mover para a nossa pasta
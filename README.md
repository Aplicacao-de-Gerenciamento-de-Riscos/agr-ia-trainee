# 🧠 Treinamento da IA - Gerenciamento de Riscos

Este repositório contém o código necessário para treinar o modelo de IA utilizado no gerenciamento de riscos. O projeto é desenvolvido em **Python** e utiliza **SQLAlchemy** para integração com o banco de dados **PostgreSQL**.

---

## 📋 Pré-requisitos

Antes de rodar o projeto, você precisa ter os seguintes itens instalados:

- 🐍 **Python 3.12** - [Download aqui](https://www.python.org/downloads/)
- 🐘 **PostgreSQL** - [Guia de instalação](https://www.postgresql.org/download/)

### Instalando as Dependências

Para instalar as dependências do projeto, execute:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Configuração do Banco de Dados

### 1. Criação do Banco de Dados PostgreSQL

Execute os seguintes comandos no PostgreSQL para criar o banco de dados e o usuário:

```sql
CREATE DATABASE agrbackend-dev;
CREATE USER usuario_ia WITH ENCRYPTED PASSWORD 'senha_ia';
GRANT ALL PRIVILEGES ON DATABASE agr_ia_train TO usuario_ia;
```

### 2. Configurar `database.py`

A configuração do banco de dados é feita no arquivo `database.py`. Edite esse arquivo para definir suas configurações de banco de dados PostgreSQL:

**Exemplo de `database.py`:**

```python
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/agrbackend-dev")
```

---

## 🚀 Treinando o Modelo

Para treinar o modelo de IA, siga os passos abaixo:

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/Aplicacao-de-Gerenciamento-de-Riscos/agr-ia-trainee.git
   cd agr-ia-trainee
   ```

2. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Certifique-se de que o banco de dados está configurado corretamente** no arquivo `database.py` e que o banco de dados PostgreSQL está rodando.

4. **Execute o script de treino:**

   ```bash
   python train_model.py
   ```

5. **Saída Esperada:**

   O modelo será treinado com os dados disponíveis e o arquivo do modelo treinado será salvo na pasta do projeto, por exemplo:

   ```
   modelo_ia.joblib
   ```

---

## 📂 Estrutura do Projeto

```
agr-ia-trainee/
│-- data/                # Pasta para os datasets
│-- models/              # Pasta para salvar os modelos treinados
│-- database.py          # Configuração do banco de dados
│-- train_model.py       # Script principal para treino do modelo
│-- requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto
```

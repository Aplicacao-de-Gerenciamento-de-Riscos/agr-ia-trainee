# data_processing.py
from sqlalchemy.orm import Session
from models import Issue, Sprint, VersionIssue, Version, Project
import pandas as pd

def load_data(db: Session):
    project_sprint_map = {}  # Mapeamento de projetos para sprints
    #O mapeamento de projetos para sprints é feito utilizando um dicionário, onde analisamos o nome das sprints e verificamos se a key do projeto está presente no nome da sprint
    #Se estiver presente, adicionamos a sprint ao projeto correspondente
    # Carregar todas as sprints e seus projetos
    sprints = db.query(Sprint).all()
    projects = db.query(Project).all()
    for sprint in sprints:
        for project in projects:
            if project.key in sprint.name:
                if project.cod_project not in project_sprint_map:
                    project_sprint_map[project.cod_project] = []
                project_sprint_map[project.cod_project].append(sprint)

    # Carregar todas as issues e suas relações com sprints, versões e projetos
    issues = (db.query(Issue)
                .join(Sprint, Issue.cod_sprint == Sprint.cod_sprint)  # Relaciona Issues com Sprints
                .join(VersionIssue, Issue.cod_issue == VersionIssue.cod_issue)  # Relaciona Issues com VersionIssues
                .join(Version, VersionIssue.cod_version == Version.cod_version)  # Relaciona VersionIssues com Versions
                .join(Project, Version.cod_project == Project.cod_project)  # Relaciona Versions com Project
                .all())

    # Converte as issues para um DataFrame do pandas
    issue_data = pd.DataFrame([{
        'cod_issue': issue.cod_issue,
        'time_original_estimate': issue.time_original_estimate,
        'timespent': issue.timespent,
        'priority': issue.priority,
        # 'status': issue.status,
        'cod_sprint': issue.cod_sprint,  # Sprint associada
        'cod_epic': issue.cod_epic,  # Épico associado
        'issuetype': issue.issuetype,  # Tipo de Issue
        #Setar o cod_project para o projeto correspondente à sprint
        'cod_project': next((project.cod_project for project in projects if any(sprint.cod_sprint == issue.cod_sprint for sprint in project_sprint_map[project.cod_project])), None),
        'cod_version': issue.versions[0].cod_version  # Versão associada
    } for issue in issues])

    # Fazer as transformações necessárias (one-hot encoding para status, priority, issuetype e cod_epic)
    issue_data = pd.get_dummies(issue_data, columns=['priority', 'issuetype', 'cod_epic'])

    # Agrupar por versão e projeto (cod_version, cod_project)
    version_data = issue_data.groupby(['cod_version', 'cod_project']).agg({
        'time_original_estimate': 'sum',  # Soma o tempo estimado para todas as issues da versão
        'timespent': 'sum',  # Soma o tempo gasto em todas as issues da versão
        # Para one-hot encoding, usamos a média (se houver pelo menos 1 ocorrência, será > 0)
        **{col: 'mean' for col in issue_data.columns if col.startswith('priority_')},
        **{col: 'mean' for col in issue_data.columns if col.startswith('status_')},
        **{col: 'mean' for col in issue_data.columns if col.startswith('issuetype_')},
        **{col: 'sum' for col in issue_data.columns if col.startswith('cod_epic_')}  # Soma ao invés de média para cod_epic
    }).reset_index()

    # Remover coluna de épico que só possuem valores em uma linha
    # Buscar colunas de épico
    epic_cols = [col for col in version_data.columns if col.startswith('cod_epic_')]

    # Iterar sobre as colunas de épico e verificar em quantas linhas o valor é > 0
    for col in epic_cols:
        if version_data[col].gt(0).sum() < 2:
            # Remover a coluna caso só tenha 1 ou 0 ocorrências
            version_data.drop(columns=[col], inplace=True)

    return version_data

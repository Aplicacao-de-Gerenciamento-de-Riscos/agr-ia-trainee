# data_processing.py
from sqlalchemy.orm import Session
from models import Issue, Sprint, VersionIssue, Version, Project
import pandas as pd

def load_data(db: Session):
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
        'time_estimate': issue.time_estimate,
        'timespent': issue.timespent,
        'priority': issue.priority,
        'status': issue.status,
        'cod_sprint': issue.cod_sprint,  # Sprint associada
        'cod_epic': issue.cod_epic,  # Épico associado
        'issuetype': issue.issuetype,  # Tipo de Issue
        'cod_project': issue.sprint.project.cod_project  # Inclui o cod_project no DataFrame
    } for issue in issues])

    # Fazer as transformações necessárias (one-hot encoding para status, priority, issuetype e cod_epic)
    issue_data = pd.get_dummies(issue_data, columns=['priority', 'status', 'issuetype', 'cod_epic'])

    # Agrupar por sprint e projeto (cod_sprint, cod_project)
    sprint_data = issue_data.groupby(['cod_sprint', 'cod_project']).agg({
        'time_estimate': 'sum',  # Soma o tempo estimado para todas as issues da sprint
        'timespent': 'sum',  # Soma o tempo gasto em todas as issues da sprint
        # Para one-hot encoding, usamos a média (se houver pelo menos 1 ocorrência, será > 0)
        **{col: 'mean' for col in issue_data.columns if col.startswith('priority_')},
        **{col: 'mean' for col in issue_data.columns if col.startswith('status_')},
        **{col: 'mean' for col in issue_data.columns if col.startswith('issuetype_')},
        **{col: 'sum' for col in issue_data.columns if col.startswith('cod_epic_')}  # Soma ao invés de média para cod_epic
    }).reset_index()

    return sprint_data

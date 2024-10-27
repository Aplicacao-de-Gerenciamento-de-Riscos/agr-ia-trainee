from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKey, TIMESTAMP, Sequence
from sqlalchemy.orm import relationship
from database import Base


# Tabela: tb_epic
class Epic(Base):
    __tablename__ = 'tb_epic'
    cod_epic = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255))
    summary = Column(String(255))
    key = Column(String(255))

    # Relacionamento com issues
    issues = relationship("Issue", back_populates="epic")


# Tabela: tb_sprint
class Sprint(Base):
    __tablename__ = 'tb_sprint'
    cod_sprint = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255))
    state = Column(String(500))
    start_date = Column(TIMESTAMP)
    end_date = Column(TIMESTAMP)
    complete_date = Column(TIMESTAMP)
    goal = Column(String(1000))

    # Relacionamento com issues
    issues = relationship("Issue", back_populates="sprint")


# Tabela: tb_component
class Component(Base):
    __tablename__ = 'tb_component'
    cod_component = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255))


# Tabela: tb_worklog
class Worklog(Base):
    __tablename__ = 'tb_worklog'
    cod_worklog = Column(BigInteger, primary_key=True, index=True, server_default=Sequence('seq_worklog').next_value())
    start_at = Column(Integer)
    max_results = Column(Integer)
    total = Column(Integer)

    # Relacionamento com issues
    issues = relationship("Issue", back_populates="worklog")

    # Relacionamento com worklog entries
    entries = relationship("WorklogEntry", back_populates="worklog")

# Tabela: tb_worklog_entry
class WorklogEntry(Base):
    __tablename__ = 'tb_worklog_entry'
    cod_worklog_entry = Column(BigInteger, primary_key=True, index=True,
                               server_default=Sequence('seq_worklog_entry').next_value())
    self_url = Column(String(255))
    author = Column(String(255))
    created = Column(TIMESTAMP)
    updated = Column(TIMESTAMP)
    time_spent = Column(String(255))
    cod_worklog = Column(BigInteger, ForeignKey('tb_worklog.cod_worklog'))

    # Relacionamento com worklog
    worklog = relationship("Worklog", back_populates="entries")


# Tabela: tb_project
class Project(Base):
    __tablename__ = 'tb_project'
    cod_project = Column(BigInteger, primary_key=True, index=True)
    key = Column(String(255))
    board_id = Column(BigInteger)

    # Relacionamento com versões
    versions = relationship("Version", back_populates="project")


# Tabela: tb_version
class Version(Base):
    __tablename__ = 'tb_version'
    cod_version = Column(BigInteger, primary_key=True, index=True)
    description = Column(String(10000))
    name = Column(String(255))
    archived = Column(Boolean)
    released = Column(Boolean)
    start_date = Column(TIMESTAMP)
    release_date = Column(TIMESTAMP)
    overdue = Column(Boolean)
    user_start_date = Column(String(255))
    user_release_date = Column(String(255))
    cod_project = Column(BigInteger, ForeignKey('tb_project.cod_project'))

    # Relacionamento com projeto
    project = relationship("Project", back_populates="versions")

    # Relacionamento com issues através de VersionIssue
    issues = relationship("Issue", secondary="tb_version_issue", back_populates="versions")


# Tabela: tb_issue
class Issue(Base):
    __tablename__ = 'tb_issue'

    cod_issue = Column(BigInteger, primary_key=True, index=True)
    key = Column(String(255))
    time_original_estimate = Column(BigInteger)
    time_estimate = Column(BigInteger)
    work_ratio = Column(BigInteger)
    cod_worklog = Column(BigInteger, ForeignKey('tb_worklog.cod_worklog'))
    status = Column(String(255))
    timespent = Column(BigInteger)
    resolution_date = Column(TIMESTAMP)
    updated = Column(TIMESTAMP)
    created = Column(TIMESTAMP)
    flagged = Column(Boolean)
    assignee = Column(String(255))
    priority = Column(String(255))
    issuetype = Column(String(255))
    summary = Column(String(255))
    cod_epic = Column(BigInteger, ForeignKey('tb_epic.cod_epic'))
    cod_sprint = Column(BigInteger, ForeignKey('tb_sprint.cod_sprint'))
    cod_parent = Column(BigInteger, ForeignKey('tb_issue.cod_issue'))

    # Relacionamentos
    epic = relationship("Epic", back_populates="issues")
    sprint = relationship("Sprint", back_populates="issues")
    worklog = relationship("Worklog", back_populates="issues")

    # Relacionamento com versões através de VersionIssue
    versions = relationship("Version", secondary="tb_version_issue", back_populates="issues")


# Tabela: tb_version_issue (tabela de relacionamento)
class VersionIssue(Base):
    __tablename__ = 'tb_version_issue'

    cod_version_issue = Column(BigInteger, primary_key=True, index=True,
                               server_default=Sequence('seq_version_issue').next_value())
    cod_version = Column(BigInteger, ForeignKey('tb_version.cod_version'))
    cod_issue = Column(BigInteger, ForeignKey('tb_issue.cod_issue'))


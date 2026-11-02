"""create_datasets_table

Revision ID: 0001
Revises:
Create Date: 2026-05-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension before creating the vector column.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        # Create as text first, then cast to vector(512) once the extension is loaded.
        sa.Column('embedding', sa.Text(), nullable=False),
        sa.Column('path', sa.String(), nullable=True),
    )

    op.execute(
        "ALTER TABLE datasets "
        "ALTER COLUMN embedding TYPE vector(512) USING embedding::vector"
    )


def downgrade() -> None:
    op.drop_table('datasets')
    op.execute("DROP EXTENSION IF EXISTS vector")

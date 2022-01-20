import pandas as pd


def get_program(row):
    if row.ADMISSION_REC_1 == 1.0:
        program = row.PROGRAM_1
    elif row.ADMISSION_REC_2 == 1.0:
        program = row.PROGRAM_2
    elif row.ADMISSION_REC_3 == 1.0:
        program = row.PROGRAM_3
    else:
        program = float('nan')
    if isinstance(program, str):
        # Take only the first number in the code of the education plan.
        program = program.split('.')[0]
    return program


def get_admission_cond(row):
    if row.ADMISSION_REC_1 == 1.0:
        admission_cond = row.ADMISSION_COND_1
    elif row.ADMISSION_REC_2 == 1.0:
        admission_cond = row.ADMISSION_COND_2
    elif row.ADMISSION_REC_3 == 1.0:
        admission_cond = row.ADMISSION_COND_3
    else:
        admission_cond = float('nan')
    return admission_cond


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    df = df.copy()

    df['PROGRAM'] = df.apply(
        func=get_program,
        axis=1,
    )
    df['ADMISSION_COND'] = df.apply(
        func=get_admission_cond,
        axis=1,
    )

    df = pd.get_dummies(df, columns=['NATIONALITY'])
    df['GENDER'] = df.GENDER.astype('category').cat.codes

    return df



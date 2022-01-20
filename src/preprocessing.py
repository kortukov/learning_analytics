import pandas as pd

from . import config


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    df = df.copy()
    df = df.drop(columns=config.COMMON_DROP_COLUMNS)

    df = drop_na(df)

    df = preprocess_program(df)
    df = preprocess_admission(df)
    df = preprocess_state_exams(df)


    df = pd.get_dummies(df, columns=["NATIONALITY"])
    df = pd.get_dummies(df, columns=["MACRO_BIRTH_PLACE"])
    df["GENDER"] = df.GENDER.astype("category").cat.codes

    return df


def preprocess_program(df: pd.DataFrame) -> pd.DataFrame:
    def get_program(row):
        if row.ADMISSION_REC_1 == 1.0:
            program = row.PROGRAM_1
        elif row.ADMISSION_REC_2 == 1.0:
            program = row.PROGRAM_2
        elif row.ADMISSION_REC_3 == 1.0:
            program = row.PROGRAM_3
        else:
            program = float("nan")
        if isinstance(program, str):
            # Take only the first number in the code of the education plan.
            program = program.split(".")[0]
        return program

    df = df.copy()
    df["PROGRAM"] = df.apply(func=get_program, axis=1)
    return df


def preprocess_admission(df: pd.DataFrame) -> pd.DataFrame:
    def get_admission_cond(row):
        if row.ADMISSION_REC_1 == 1.0:
            admission_cond = row.ADMISSION_COND_1
        elif row.ADMISSION_REC_2 == 1.0:
            admission_cond = row.ADMISSION_COND_2
        elif row.ADMISSION_REC_3 == 1.0:
            admission_cond = row.ADMISSION_COND_3
        else:
            admission_cond = float("nan")
        return admission_cond

    df = df.copy()
    df["ADMISSION_COND"] = df.apply(func=get_admission_cond, axis=1)
    return df


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["START_YEAR"])
    return df


def preprocess_state_exams(df: pd.DataFrame) -> pd.DataFrame:
    exams_fillna_dict = (
        df[["START_YEAR"] + config.STATE_EXAMS]
        .groupby("START_YEAR")
        .median()
        .fillna(method="bfill")
        .to_dict()
    )
    for exam in config.STATE_EXAMS:
        (df[exam], df[exam + "_BIN"]) = zip(
            *df.apply(
                lambda x: (exams_fillna_dict[exam][x["START_YEAR"]], 0)
                if (x[exam] != x[exam])
                else (x[exam], 1),
                axis=1,
            )
        )

    return df

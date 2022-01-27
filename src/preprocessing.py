import pandas as pd

from . import config


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    df = df.copy()
    df = df.drop(columns=config.COMMON_DROP_COLUMNS)

    df = drop_na(df)

    df = preprocess_program(df)
    df = preprocess_admission(df)
    df = df.drop(columns=[f"ADMISSION_REC_{i}" for i in range(1, 4)])

    df = preprocess_state_exams_scores(df)
    df = preprocess_state_exams(df)
    df = preprocess_admission_type(df)
    df = preprocess_subject_prize_level(df)
    df = preprocess_state_exams_sum(df)

    df = pd.get_dummies(df, columns=["NATIONALITY"])
    df = pd.get_dummies(df, columns=["MACRO_BIRTH_PLACE"])
    df = pd.get_dummies(df, columns=["SCHOLARSHIP_TYPE"], dummy_na=True)
    df = pd.get_dummies(df, columns=["DORMITORY_FOR_STUDY"], dummy_na=True)

    df["AGE_AT_START"] = df["AGE_AT_START"].fillna(df["AGE_AT_START"].median())
    df["GENDER"] = df.GENDER.astype("category").cat.codes
    df["DROPPED_TARGET"] = df["DROPPED_TARGET"].apply(
        lambda x: 0 if x == 1 else (1 if x == 0 else x)
    )
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
    df = pd.get_dummies(df, columns=["PROGRAM"])
    df = df.drop(columns=[f"PROGRAM_{i}" for i in range(1, 4)])
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
    df = pd.get_dummies(df, columns=["ADMISSION_COND"])

    df = df.drop(columns=[f"ADMISSION_COND_{i}" for i in range(1, 4)])

    return df


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["START_YEAR"])
    return df


def preprocess_state_exams_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def preprocess_state_exams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dummies = [
        pd.get_dummies(df, columns=[f"EXAM_SUBJECT_{i}"], prefix="", prefix_sep="", dummy_na=True)
        for i in range(1, 4)
    ]
    for exam in set(dummies[0].columns):
        for col in dummies:
            if exam in col.columns:
                df[exam] = col[exam]
    df = df.drop(columns=[f"EXAM_SUBJECT_{i}" for i in range(1, 4)])
    return df


def preprocess_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    df["ADMISSION_TYPE"] = df.apply(
        lambda x: "ЕГЭ"
        if (x["ADMISSION_TYPE"] != x["ADMISSION_TYPE"])
        and (x["ADMITTED_EXAMS_SUM"] == x["ADMITTED_EXAMS_SUM"])
        else x["ADMISSION_TYPE"],
        axis=1,
    )
    df = pd.get_dummies(df, columns=["ADMISSION_TYPE"], dummy_na=True)

    return df


def preprocess_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    df["ADMISSION_TYPE"] = df.apply(
        lambda x: "ЕГЭ"
        if (x["ADMISSION_TYPE"] != x["ADMISSION_TYPE"])
        and (x["ADMITTED_EXAMS_SUM"] == x["ADMITTED_EXAMS_SUM"])
        else x["ADMISSION_TYPE"],
        axis=1,
    )
    df = pd.get_dummies(df, columns=["ADMISSION_TYPE"], dummy_na=True)

    return df


def preprocess_subject_prize_level(df: pd.DataFrame) -> pd.DataFrame:
    df["ADMITTED_SUBJECT_PRIZE_LEVEL"] = df["ADMITTED_SUBJECT_PRIZE_LEVEL"].apply(
        lambda x: x if x != 4 else 3
    )

    df["ADMITTED_SUBJECT_PRIZE_LEVEL"] = df.apply(
        lambda x: 4
        if (x["ADMITTED_SUBJECT_PRIZE_LEVEL"] != x["ADMITTED_SUBJECT_PRIZE_LEVEL"])
        and (x["ADMITTED_EXAMS_SUM"] == x["ADMITTED_EXAMS_SUM"])
        else x["ADMITTED_SUBJECT_PRIZE_LEVEL"],
        axis=1,
    )
    df = pd.get_dummies(df, columns=["ADMITTED_SUBJECT_PRIZE_LEVEL"], dummy_na=True)

    return df


def preprocess_state_exams_sum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exam_fillna_dict = (
        df[["START_YEAR", "ADMITTED_EXAMS_SUM"]]
        .groupby("START_YEAR")
        .median()
        .fillna(method="bfill")
        .to_dict()
    )
    (df["ADMITTED_EXAMS_SUM"], df["ADMITTED_EXAMS_SUM_BIN"]) = zip(
        *df.apply(
            lambda x: (exam_fillna_dict["ADMITTED_EXAMS_SUM"][x["START_YEAR"]], 0)
            if (x["ADMITTED_EXAMS_SUM"] != x["ADMITTED_EXAMS_SUM"])
            else (x["ADMITTED_EXAMS_SUM"], 1),
            axis=1,
        )
    )

    return df
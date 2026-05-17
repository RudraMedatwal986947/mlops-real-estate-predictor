from pydantic import BaseModel


class HouseData(BaseModel):

    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int
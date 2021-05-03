import requests
from numpy import NaN
import pandas as pd


def median_income_acs(state="06", county="037", city="Los Angeles"):
    """Retrieves 5-year estimates of median block level household income
    from the U.S. Census Bureau API and writes to the current directory."""
    # ACS 2019 5-year data API url
    url = "https://api.census.gov/data/2019/acs/acs5"

    # Median household income variable name
    variable = "B19013_001E"

    # All blocks in CA (06), LA County (037)
    geo = f"&in=state:{state}+county:{county}+tract:*"

    # API request
    response = requests.get(f"{url}?get=NAME,{variable}&for=block%20group:*{geo}")

    if response.ok is True:
        income = pd.DataFrame(response.json()[1:], columns=response.json()[0])
        income[variable] = pd.to_numeric(income[variable])
        income.loc[income[variable] < 0, variable] = NaN
        # Write response to csv file
        income.to_csv(f"data/acs/{city}/{state}{county}_block_household_income.csv",
                      index=False)
        print(response)
    else:
        print(response)


def median_income_census_reporter(state="06", place="44000", city="Los Angeles"):
    """Retrieves 5-year estimates of median block level household income
    from Census Reporter and writes to the current directory."""
    # Census Reporter API url for ACS 2019 5-year
    url = "https://api.censusreporter.org/1.0/data/show/acs2019_5yr"

    # Median Household income variable name
    variable = "B19013"

    # Block level (140), for place (160), geographic component(00)
    # ,CA (06), LA City (44000)
    geo = f"150|16000US{state}{place}"

    # API request
    response = requests.get(f"{url}?table_ids={variable}&geo_ids={geo}")

    if response.ok is True:
        content = response.json()
        income = pd.DataFrame(index=range(len(content["data"])),
                              columns=["geoid", "name", "error", "estimate"])
        for i, geoid in enumerate(content["data"].keys()):
            income.loc[i, "geoid"] = geoid
            income.loc[i, "name"] = content["geography"][geoid]["name"]
            income.loc[i, "error"] = content["data"][geoid]["B19013"]["error"]["B19013001"]
            income.loc[i, "estimate"] = content["data"][geoid]["B19013"]["estimate"]["B19013001"]
        # Write response to csv file
        income.to_csv(f"data/acs/{city}/16000US{state}{place}_block_household_income.csv",
                      index=False)
        print(response)
    else:
        print(response)


if __name__ == "__main__":
    median_income_census_reporter()

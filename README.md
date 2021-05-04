# asxtrade

![tox workflow](https://github.com/ozacas/asxtrade/actions/workflows/python-package.yml/badge.svg)

Python3 based ASX data download and web application with basic features:

 * ability to search by sector, keyword, movement, dividend yield or other attributes

 * watchlist and sector stock lists, sortable by key metrics (eps, pe, daily change etc.)

 * historical company metrics (cashflow, earnings, balance sheet) using [YFinance](https://pypi.org/project/yfinance/)

 * graphs of datasets over 12 months (or whatever data is available)

 * portfolio optimisation using [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html)

 * Market and per-sector performance incl. top 20/bottom 20

 * Virtual portfolios with trend visualisations, profit/loss by stock

 * Stock point-scoring by rules over time, based on their price performance

 * Visualisations provided by [plotnine](https://github.com/has2k1/plotnine) and [matplotlib](https://github.com/matplotlib/matplotlib)

 ## Disclaimer

This software is provided to you "as-is" and without
warranty of any kind, express, implied or otherwise, including without
limitation, any warranty of fitness for a particular purpose. In no event shall
the author(s) be liable to you or anyone else for any direct, special, incidental,
indirect or consequential damages of any kind, or any damages whatsoever,
including without limitation, loss of profit, loss of use, savings or revenue,
or the claims of third parties, whether or not the author(s) have
been advised of the possibility of such loss, however caused and on any theory
of liability, arising out of or in connection with the possession, use or
performance of this software.


 ## System Requirements

  * Python 3.8

  * Django

  * Djongo (python MongoDB for Django apps)

  * Other requirements documented in [requirements.txt](./requirements.txt)


## Installation

### Pre-requisite software installation

On the system to run the website, you'll need to install required software:
~~~~
git clone https://github.com/ozacas/asxtrade.git
cd asxtrade
sudo pip3 install -r requirements.txt
cd src/viewer # for setting up the website
~~~~

### Setup MongoDB database server

Next, you'll want to setup a database to store the ASX data: instructions for this can be found at the [MongoDB website](https://docs.mongodb.com/manual/administration/install-community/)

### Setup the website and superuser for administrating the site

~~~~
python3 manage.py migrate

python3 manage.py createsuperuser

export MPLBACKEND=Agg # recommended: all testing is done with this matplotlib backend plugin to integrate with Django server
python3 manage.py runserver # run on local dev. host
~~~~

### Installing data

  This application only works with daily data fetched after 4pm each trading day from the ASX website. It will take several hours per run. After the run, you must update the current month's pre-pivoted market data cache. The setup and daily run process is similar to:
  
  ~~~~
  # setup database password to use
  export PASSWORD=<your db password here>

  # run periodically (eg. quarterly) and at setup to update list of securities/companies/fundamentals
  python3 src/asxtrade.py --want-companies
  python3 src/asxtrade.py --want-isin
  python3 src/asxtrade.py --want-details

  # daily: fetch prices from ASX - configuration in config.json. Run after trading close each trading day
  python3 src/asxtrade.py --want-prices

  # daily: update cached data for the specified month (1..12) - needed for website operation
  python3 persist_dataframes.py --month 3 --year 2021 --status INCOMPLETE|FINAL --dbpassword '$PASSWORD'
  ~~~~
  
This new data may not appear in the application until the cache entries expire (or alternatively you can restart the server).

  Existing data ready to import into mongodb v4.4 can be fetched from [github large file storage](https://github.com/ozacas/asxtrade/raw/master/data/asxtrade.20210306.bson.gz) using [mongorestore](https://docs.mongodb.com/database-tools/mongorestore/). This data covers the daily data from July 2020 thru March 2021, although ETF data covers a smaller period due to missing code.

## Features

 | Feature             | Thumbnail Picture |
 |:--------------------|------------------:|
 | Portfolio watchlist | ![Pic](https://user-images.githubusercontent.com/11968760/91777314-da1bdb00-ec32-11ea-929e-66a1befc0d90.png#thumbnail)|
 | Stock view | ![Pic](https://user-images.githubusercontent.com/11968760/91777703-ed7b7600-ec33-11ea-87bf-b647033ed06f.png)|
 | Market sentiment | ![Pic](https://user-images.githubusercontent.com/11968760/91778464-e48ba400-ec35-11ea-9b47-413601da6fd8.png#thumbnail)|
 | Performance by sector | ![Pic](https://user-images.githubusercontent.com/11968760/110228446-6a760800-7f55-11eb-9041-786e6d145817.png#thumbnail)|
 | Portfolio optimisation | ![Pic](https://user-images.githubusercontent.com/11968760/110228663-e7ee4800-7f56-11eb-8b7d-edd3a09d7b29.png#thumbnail)
 | World Bank API data: more than 10000 free datasets in over twenty subject areas | ![Pic](https://user-images.githubusercontent.com/11968760/115988232-00481e00-a5fc-11eb-9ab7-afa3a5365cb2.png)|


## Testing

Tox is used to drive pytest to execute the unit test. Django `manage.py test` is not used. Github Actions is then used to invoke tox for CI/CD for the application and unit tests.

HTML unit test coverage reports can be produced using something like (from the root of the local repo):

~~~~
$ export PYTHONPATH=`pwd`/src/viewer
$ pytest -c pytest.ini --cov-report html --cov-config=.coveragerc --cov=src .
$ ls htmlcov
# open htmlcov/index.html in your favourite browser
~~~~

## Debugging

Use of the Django Debug Toolbar is strongly recommended, but it does have a known limitation which prevents usage during normal operation due to the use of pandas within asxtrade: see <https://github.com/jazzband/django-debug-toolbar/issues/1053> for details. For this reason, the MIDDLEWARE setting for
the toolbar is commented out, except when needed.

### Macroeconomic datasets: World Bank (experimental)

Recent releases include support for API data integration using the excellent [wbdata](https://pypi.org/project/wbdata/), persisting the data to parquet format in the mongo database for the app. A utility script is provided to download this data: but it works very slowly to be nice to the Internet (only 2 datasets per minute). At that rate
it would take over a week to download the entire dataset of nearly 18000 dataframes! Only year-based datasets are currently supported.

~~~~
python3 ingest_worldbank_datasets.py --help
~~~~

This program builds an inverted index of the WorldBank data, to speed searching via `Show -> World Bank` data on the left-side. Three different analyses are possible:

 1. A single metric for a single country, each year
 2. A single metric for several countries
 3. Many metrics for a single country

Each analysis comes with its own visualisations and features. More to come...

### Ingest data for ASX company financials (experimental)

It is now possible, but entirely optional, to add company financial performance metrics (earnings, cashflow, balance sheet etc.) to an existing database:

~~~~
python3 ingest_financials.py --help
~~~~

Ingestion of financial metrics should be run when companies update the market on their performance (eg. mid-year and year-end). To be nice to Yahoo Finance API, it will take a couple of days to perform a complete run.

It has [been observed](https://github.com/ranaroussi/yfinance/issues/250) that in some cases the data ingested can be wrong, 
particularly for ASX stocks. Use at own risk. This function is therefore considered experimental and pages using this data are labelled with a warning.
If data is not ingested, then the 'Show financial performance' button on stock views (as well as data download) will 404. This data is not available for ETFs at this time. It is planned to support search for companies by a metric. 

### Macroeconomic datasets: Australian Bureau of Statistics (experimental)

TODO FIXME... still waiting for an API key from the government.
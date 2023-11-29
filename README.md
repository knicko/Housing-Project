{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "739f2a6f",
   "metadata": {},
   "source": [
    "# 1. Problem Statement:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "668fb6e3",
   "metadata": {},
   "source": [
    "There is not a clear way to value homes in the market. I am a Data Scientist contracted by the Appraiser Qualifications Board. I am in charge of providing developers with tips on how to price their homes and what could be improved in order to push the economy forward. I am pitching this slideshow to real estate developers who plan on building new properties in the state of Iowa.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2733f57b",
   "metadata": {},
   "source": [
    "# 2. Description of Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2fdaae2b",
   "metadata": {},
   "source": [
    "The dataset that I used consist of a train.csv and test.csv. Both datasets had 80 columns to work with."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7cd68216",
   "metadata": {},
   "source": [
    "The features consisted of:\n",
    "'Id', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',\n",
    "       'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',\n",
    "       'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',\n",
    "       'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual',\n",
    "       'Overall Cond', 'Year Built', 'Year Remod/Add', 'Roof Style',\n",
    "       'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',\n",
    "       'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',\n",
    "       'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',\n",
    "       'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',\n",
    "       'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF',\n",
    "       '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',\n",
    "       'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',\n",
    "       'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional',\n",
    "       'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Yr Blt',\n",
    "       'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual',\n",
    "       'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF',\n",
    "       'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC',\n",
    "       'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type',\n",
    "       'SalePrice'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "92e75d12",
   "metadata": {},
   "source": [
    "Of all of these, I decided to drop only 'PID' and Convert MS Subclass to string."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cc17ca2",
   "metadata": {},
   "source": [
    "The data was found at: https://www.kaggle.com/competitions/project-2-regression-challenge-123/data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e9a1ead",
   "metadata": {},
   "source": [
    "Target: Linear Regression, Ridge Regression, and Lasso Model\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07a5a1b4",
   "metadata": {},
   "source": [
    "Data Dictionary:\n",
    "\n",
    "| Feature | Type   | Dataset | Description  |\n",
    "|---------|--------|---------|--------------|\n",
    "|   total_data     |   table|    train.csv    |  train.csv renamed into 'total_data'       |\n",
    "|  known_list     |   list|    train.csv    |  Columns in train & test that are NON-Numeric & NA means something       |\n",
    "|   knn_impute     |   function|    train.csv/test.csv    |  function that takes in dataframe, and target, then returns new dataframe with filled nums using knnRegressors       |\n",
    "|   total_data['total_bathrooms']     |   column|    train.csv    |  column that adds all bathrooms together       |\n",
    "|    total_data['total_home_quality']    |   column|    train.csv    |  column that adds all quality metrics together        |\n",
    "|   total_data_dummy    |   table|    train.csv    | Create Dummies for Categoricals      |\n",
    "|   preds     |   array|   train.csv    |  Predictions from Linear Regression, Ridge Regression, and Lasso Model      |\n",
    "|   top5     |   dictionary|    train.csv    |  dictionary that represents important factors contributing to Sale Price with values representing percentages. Highest = Correlated with Sale Price       |\n",
    "|   mat5    |   dictionary|    train.csv    |  dictionary that represents TOP MATERIAL factors directly related to Sale Price|\n",
    "|   avg_price     |   function|    train.csv    |  function that takes in a column and returns the mean of the Sale Price for all       |\n",
    "|   avg_sqft     |   float|    train.csv    |  average square foot of a house in Ames, Iowa       |\n",
    "|   avglot_sqft     | float|    train.csv    |  average square foot of a lot in Ames, Iowa       |\n",
    "|   avg_msn     | float|    train.csv    |  average square foot of a masonry vaneer in Ames, Iowa     |\n",
    "|   final_submit     | table|    final_submit.csv    |  final submission with predictions      |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe9a2508",
   "metadata": {},
   "source": [
    "EDA: \n",
    "1. Correlation Map with top features correlating to sale prices\n",
    "top_corr_features = correlation.index[abs(correlation[\"SalePrice\"]) > 0.5]\n",
    "plt.subplots(figsize=(12,9))\n",
    "sns.heatmap(train[top_corr_features].corr(),annot=True,cmap=\"Accent\")\n",
    "\n",
    "2. Distribution Chart that shows sales are RIGHT skewed. \n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.distplot(train['SalePrice'])\n",
    "plt.show()\n",
    "\n",
    "3. Outliers identified in Lot_area vs Sale_Price\n",
    "fig, ax = plt.subplots()\n",
    "ax.scatter(x = train['Lot Area'], y = train['SalePrice'])\n",
    "plt.ylabel('SalePrice')\n",
    "plt.xlabel('GrLivArea')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b4c7fee",
   "metadata": {},
   "source": [
    "# 3. I fitted 4 models, but didn't like Elastic Net"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da2d72e7",
   "metadata": {},
   "source": [
    "======= LR =======\n",
    "\n",
    "0.9461986048529202\n",
    "\n",
    "0.6147163362891986\n",
    "\n",
    "\n",
    "\n",
    "===== Ridge ======\n",
    "\n",
    "0.9316969711857228\n",
    "\n",
    "0.7784409211368675\n",
    "\n",
    "\n",
    "\n",
    "===== Lasso ======\n",
    "\n",
    "0.9384261649374445\n",
    "\n",
    "0.7670325488790939\n",
    "\n",
    "\n",
    "\n",
    "=== ElasticNet ===\n",
    "\n",
    "0.8969783605015145\n",
    "\n",
    "0.7451396076366255"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f671a173",
   "metadata": {},
   "source": [
    "# 4. Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2d74463",
   "metadata": {},
   "source": [
    "I have came to the conclusion that the more features I toss into the model, the better it performs. I initially tried to fit the models with zero'd in values for nulls. But I went back and filled some in with mean and knn. This helped out a lot. I then tried to see which fitting operated the best. It seems like Ridge and Lasso are hand in hand. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1e3448f",
   "metadata": {},
   "source": [
    "Answer: With my findings, every small feature needs to be taken into account when appraising the value of a home. The price of a home will never be true value unless more data is collected. The biggest factors when pricing a home is Square Feet, Quality, Garage Space, Neighborhood, and Number of Bedrooms."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import streamlit as st
import pandas as pd
from yahoo_fin import stock_info as si
from yahoo_fin import news
import datetime as dt
import numpy as np
import datetime
import time as tm
import altair as alt

#import tensorflow as tf
#from sklearn.preprocessing import MinMaxScaler
#from collections import deque

# AI
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout

NAN = np.nan

def get_integer(number):

	number_lst = list(number.strip())
	if number == 'NaN':
		return NAN
	if number_lst[-1] == 'T':
		number_lst.pop(-1)
		number_lst.append((15-(len(number_lst)-number_lst.index('.'))-2)*'0')
		number_lst.pop(number_lst.index('.'))
		return int(''.join(number_lst))
	if number_lst[-1] == 'B':
		number_lst.pop(-1)
		number_lst.append((12-(len(number_lst)-number_lst.index('.'))-2)*'0')
		number_lst.pop(number_lst.index('.'))
		return int(''.join(number_lst))
	if number_lst[-1] == 'M':
		number_lst.pop(-1)
		number_lst.append((9-(len(number_lst)-number_lst.index('.'))-2)*'0')
		number_lst.pop(number_lst.index('.'))
		return int(''.join(number_lst))

def isNaN(num):
	return not isinstance(num, pd.DataFrame) and num != num

def not_found_wrapper(loc, name, idx=0):
	try:
		return loc[name][0]
	except:
		return NAN

class Company:

	def __init__(self, ticker, num_years):

		self.ticker = ticker
		try:
			price_df = si.get_data(ticker, dt.datetime.now()-dt.timedelta(days=num_years*365), dt.datetime.date(dt.datetime.now()))
			overview_df = si.get_stats(ticker)
		except:
			self.ticker = NAN
			return

		overview_df = overview_df.set_index('Attribute')
		overview_dict = si.get_quote_table(ticker)

		try:
			income_statement = si.get_income_statement(ticker)
		except:
			income_statement = NAN

		try:
			balance_sheet = si.get_balance_sheet(ticker)
		except:
			balance_sheet = NAN

		try:
			cash_flows = si.get_cash_flow(ticker)
		except:
			cash_flows = NAN

		self.year_end	   = not_found_wrapper(overview_df.loc, 'Fiscal Year Ends')

		if 'Market Cap' in overview_dict:
			self.market_cap	   = get_integer(overview_dict['Market Cap'])
			self.market_cap_cs = '{:,d}'.format(int(self.market_cap))
		else:
			self.market_cap	   = NAN
			self.market_cap_cs = NAN

		self.prices		   = price_df['adjclose'].rename(ticker)

		if not isNaN(income_statement):
			self.sales		  = not_found_wrapper(income_statement.loc, 'totalRevenue')
			self.gross_profit = not_found_wrapper(income_statement.loc, 'grossProfit')
			self.ebit		  = not_found_wrapper(income_statement.loc, 'ebit')
			self.interest	  = -not_found_wrapper(income_statement.loc, 'interestExpense')
			self.net_profit   = not_found_wrapper(income_statement.loc, 'netIncome')
		else:
			self.sales = NAN
			self.gross_profit = NAN
			self.ebit = NAN
			self.interest = NAN
			self.net_profit = NAN

		if not isNaN(balance_sheet):

			self.assets	      = not_found_wrapper(balance_sheet.loc, 'totalAssets')
			self.currenta	  = not_found_wrapper(balance_sheet.loc, 'totalCurrentAssets')
			self.currentl	  = not_found_wrapper(balance_sheet.loc, 'totalCurrentLiabilities')
			self.working_cap  = self.currenta - self.currentl

			self.debt = not_found_wrapper(balance_sheet.loc,'longTermDebt')
			std = not_found_wrapper(balance_sheet.loc, 'shortLongTermDebt')
			if not isNaN(std):
				self.debt += std

			self.cash = not_found_wrapper(balance_sheet.loc,'cash')
			self.inventory = not_found_wrapper(balance_sheet.loc,'inventory')
			self.receivables = not_found_wrapper(balance_sheet.loc,'netReceivables')
			self.payables = not_found_wrapper(balance_sheet.loc,'accountsPayable')
			self.equity = not_found_wrapper(balance_sheet.loc,'totalStockholderEquity')

		else:
			self.assets	      = NAN
			self.currenta	  = NAN
			self.currentl	  = NAN
			self.working_cap  = NAN
			self.debt         = NAN
			self.cash         = NAN
			self.inventory    = NAN
			self.receivables  = NAN
			self.payables     = NAN
			self.equity       = NAN


		self.net_debt = self.debt - self.cash
		self.ev     = self.market_cap + self.net_debt

		if isNaN(self.ev):
			self.ev_cs  = 'NaN'
		else:
			self.ev_cs  = '{:,d}'.format(int(self.ev))

		if not isNaN(cash_flows):
			self.operating_cf = not_found_wrapper(cash_flows.loc,'totalCashFromOperatingActivities')
			self.investing_cf = not_found_wrapper(cash_flows.loc,'totalCashflowsFromInvestingActivities')
			self.financing_cf = not_found_wrapper(cash_flows.loc,'totalCashFromFinancingActivities')
			self.capex        = -not_found_wrapper(cash_flows.loc,'capitalExpenditures')
		else:
			self.operating_cf = NAN
			self.investing_cf = NAN
			self.financing_cf = NAN
			self.capex        = NAN

		self.free_cash_flow = self.operating_cf - self.capex

		try:
			self.dividend_history = si.get_dividends(self.ticker, index_as_date=True)['dividend'].rename(ticker)
		except:
			self.dividend_history = pd.Series()

	def get_overview(self):
		self.price_earnings_ratio = self.market_cap/self.net_profit
		self.ev_sales_ratio = self.ev/self.sales
		self.overview_dict = { 'Values' : [self.ev_cs, self.market_cap_cs, self.ev_sales_ratio, self.price_earnings_ratio] }

	def get_profit_margins(self):
		self.gross_margin = self.gross_profit/self.sales
		self.operating_margin = self.ebit/self.sales
		self.net_margin = self.net_profit/self.sales
		self.profit_margin_dict = { 'Values' : [self.gross_margin, self.operating_margin, self.net_margin] }

	def get_liquidity_ratios(self):
		self.current_ratio = self.currenta/self.currentl
		self.quick_ratio = (self.currenta - self.inventory)/self.currentl
		self.cash_ratio = self.cash/self.currentl
		self.liquidity_ratio_dict = { 'Values' : [self.current_ratio, self.quick_ratio, self.cash_ratio] }

	def get_leverage_ratios(self):
		self.debt_ratio = self.debt/self.assets
		self.debt_equity_ratio = self.debt/self.equity
		self.interest_coverage_ratio = self.ebit / self.interest
		self.leverage_ratio_dict = { 'Values' : [self.debt_ratio, self.debt_equity_ratio, self.interest_coverage_ratio] }

	def get_efficiency_ratios(self):
		self.asset_turnover = self.sales/self.assets
		self.receivables_turnover = self.sales/self.receivables
		self.inventory_turnover = (self.sales-self.gross_profit)/self.inventory
		self.efficiency_ratio_dict = { 'Values' : [self.asset_turnover, self.receivables_turnover, self.inventory_turnover] }

	def get_all(self):
		self.get_overview()
		self.get_profit_margins()
		self.get_liquidity_ratios()
		self.get_leverage_ratios()
		self.get_efficiency_ratios()

st.title('Financial Dashboard')

#date          = st.sidebar.date_input('start date', datetime.date(2011,1,1))
num_years     = int(st.sidebar.number_input('Enter number of years to backtrack', min_value=3, value=10))
ticker_input  = st.sidebar.text_input('Please enter your company ticker:', value="QAN.AX BHP.AX HVN.AX VAS.AX")
search_button = st.sidebar.button('Search')
startdate     = dt.datetime.now()-dt.timedelta(days=num_years*365)

if search_button:

	Multiple_stock = ',' in ticker_input

	with st.spinner('Loading companies information ...'):

		if ',' in ticker_input:
			companies = [ (c.strip(), Company(c.strip(), num_years)) for c in ticker_input.split(',') ]
		else:
			companies = [ (c.strip(), Company(c.strip(), num_years)) for c in ticker_input.split() ]

		companies = [ (ticker, c) for ticker, c in companies if not isNaN(c.ticker) ]
		_         = [ c.get_all() for ticker, c in companies ]

	if len(companies) > 0:

		overview_index = ['Enterprise value', 'Market cap', 'EV/sales ratio', 'P/E ratio']

		with st.spinner('Loading data from Yahoo Finance ...'):
			companies_overview_df = [ pd.DataFrame(c.overview_dict, index = overview_index) for ticker, c in companies ]

		st.header('Stock price trend')
		st.line_chart(pd.concat([ c.prices for ticker, c in companies ], axis=1))
		st.header('Stock price trend (normalized)')
		st.line_chart(pd.concat([ c.prices.div(list(c.prices)[0]) for ticker, c in companies ], axis=1))
		#st.table(overview_df)

		with st.expander('Dividend history'):

			plot_df = pd.concat([ c.dividend_history for ticker, c in companies ], axis=1, join='outer').reset_index()

			chart_df = plot_df.melt('index', var_name='Ticker', value_name='Dividend').dropna()

			chart_df = chart_df[chart_df["index"] >= startdate]
			chart_df["index"] = pd.to_datetime(chart_df["index"]).dt.strftime('%Y-%m-%d')

			chart = alt.Chart(chart_df).mark_bar().encode(
				alt.X('yearquarter(index):T'),
				y='Dividend:Q',
				color='Ticker:N',
			)

			st.altair_chart(chart, use_container_width=True)

			for ticker, c in companies:
				if len(c.dividend_history) == 0:
					continue
				plot_df = c.dividend_history
				plot_df = plot_df.loc[startdate:]
				st.write(f"Dividend chart for {ticker}")
				st.bar_chart(data=plot_df, x=None, use_container_width=True)

		with st.expander('Profit margins'):

			profit_margin_index = ['Gross margin', 'Operating margin', 'Net margin']

			plot_df = pd.concat([ pd.DataFrame(c.profit_margin_dict, index = profit_margin_index).rename(columns={'Values': ticker}) for ticker, c in companies ], axis=1)
			st.table(plot_df.dropna(how='all', axis=1))

		with st.expander('Liquidity ratios'):

			liquidity_ratio_index = ['Current ratio', 'Quick ratio', 'Cash ratio']

			plot_df = pd.concat([ pd.DataFrame(c.liquidity_ratio_dict, index = liquidity_ratio_index).rename(columns={'Values': ticker}) for ticker, c in companies ], axis=1)
			st.table(plot_df.dropna(how='all', axis=1))

		with st.expander('Leverage ratios'):

			leverage_ratio_index = ['Debt/total assets ratio', 'Debt/equity ratio', 'Interest coverage ratio']

			plot_df = pd.concat([ pd.DataFrame(c.leverage_ratio_dict, index = leverage_ratio_index).rename(columns={'Values': ticker}) for ticker, c in companies ], axis=1)
			st.table(plot_df.dropna(how='all', axis=1))

		with st.expander('Efficiency ratios'):

			efficiency_ratio_index = ['Asset turnover', 'Receivables turnover', 'Inventory turnover']

			plot_df = pd.concat([ pd.DataFrame(c.efficiency_ratio_dict, index = efficiency_ratio_index).rename(columns={'Values': ticker}) for ticker, c in companies ], axis=1)
			st.table(plot_df.dropna(how='all', axis=1))

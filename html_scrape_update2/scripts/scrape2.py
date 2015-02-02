#scrape2.py

from bs4 import BeautifulSoup
import requests
import csv

url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"

r = requests.get(url)
soup = BeautifulSoup(r.content)

###Print commands for exploring the html page.
#print soup.title
#print soup.title.name
#print soup.a
#print soup.p
#print soup.get_text()
#print soup.head
#print(soup.prettify())
#print soup('table')[6]

###Attempt to extract data by "tr" tag
#link=soup.find_all("tr")
#for link in soup.find_all("tr"):
   #print link.text, link.get("tr") 
#node=soup.find_all("td")    
#for node in soup.findAll('td'):
   #print ''.join(node.findAll(text=True))

###Attempt to extract data by "table" tag
#script=soup.find_all("table")
#CDATA = str(script[0].get_text())
#print CDATA

###Another attmept to extract the data
tables = soup.find_all('table')
res = tables[6].get_text()

###Attempt to write out data to csv, only prints headers and not the data from the table.
#country = "Country or area"
#year ="Year"
#total="Total"
#men ="Men"
#women ="Women"

#f=csv.writer(open("res.csv", "w"))
#f.writerow([country, year, total, men, women])
#rowout = (country, year, total, men, women)

'''html scrape summary''
#In this exercise various BeautifulSoup methods were used to explore and extract data (stored in a table) from a website. The results from each of these 
#attempts resulted in generally the same output and format. The cleanest output resulted from: #tables = soup.find_all('table')...
# Various approaches were used to write the data to a csv file but this didn't work and the only headers were written. The data were copied to a csv file directly from the terminal window.

This site was able to extract the data and output the table to a csv directly which was useful  to verify whether the all of thee data was extracted and preserved via BeautifulSoup.
#http://www.convertcsv.com/html-table-to-csv.htm 
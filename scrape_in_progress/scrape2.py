#scrape2.py

from bs4 import BeautifulSoup
import requests
import csv

url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"
#http://www.convertcsv.com/html-table-to-csv.htm #an easier way! or download csv table directly from link  or copy from termianl window and paste in excel/csv


r = requests.get(url)
soup = BeautifulSoup(r.content)

#exploring the html
#print soup.title
#print soup.title.name
#print soup.a
#print soup.p
#print soup.get_text()
#print soup.head
#print soup.title
#print(soup.prettify())
#print soup('table')[6]
#print(soup.get_text())

#link=soup.find_all("tr")
#for link in soup.find_all("tr"):
    #print link.text, link.get("tr") 
#node=soup.find_all("td")    
#for node in soup.findAll('td'):
   #print ''.join(node.findAll(text=True))

#script=soup.find_all("table")
#CDATA = str(script[0].get_text())
#print CDATA
#href = soup.find_all("table")#this can be repeated for Year, Total, Men, Women
#Country = str(href[0].get_text())
#print Country 

#href = soup.find_all("table")#this can be repeated for Total, Men, Women
#Year = str(href[0].get_text())
#print Year

tables = soup.find_all('table')
print tables[6].get_text()

Country = "Country_or area"
Y="Year"
T="Total"
M="Men"
W="Women"

f=csv.writer(open("res.csv", "w"))#attempt to write out data to csv
#f.writerow([Country, Y, T, M, W])
#rowout = (Country, Y, T, M, W)
with open("res.csv", "w") as f:
	w=csv.writer(f)
	w.writerow("Country", "Y", "T", "M", "W")
	rowout = (Country, Y, T, M, W)
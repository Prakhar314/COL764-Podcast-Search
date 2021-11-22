import bs4

def convertToTsv(filename,qfield):
    tsv = open(filename.split('.')[0]+"_"+qfield+".tsv",'w')
    with open(filename,'r') as f:
        file = bs4.BeautifulSoup(f.read(), "lxml")
        topics = file.find_all("topic")
        for t in topics:
            num = t.find("num").text.strip()
            que = t.find(qfield).text.strip()
            tsv.write(f'{num}\t{que}\n')
    tsv.close()

if __name__ == "__main__":
    import sys
    convertToTsv(sys.argv[1],"query")
    convertToTsv(sys.argv[1],"description")



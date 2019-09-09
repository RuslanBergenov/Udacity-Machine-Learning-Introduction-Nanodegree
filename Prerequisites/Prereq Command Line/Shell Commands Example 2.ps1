pwd

cd C:\Users\Ruslan\Documents

# list all files with a specific extension
ls *.docx


cd 'C:\Users\Ruslan\Documents\Projects\RD\Analytics Edge\Unit 2 Linear Regression\Datasets and Graphs'
ls *.csv;ls *.png

ls climate* # list files by name

ls AllWine*


cd 'C:\Users\Ruslan\Documents\TEMP SPACE'

mkdir 'NEW' #create folder
mv *.jpg NEW # move. What are you moving? Where are you moving it to? # move jpg files into new

ls NEW


mv NEW\*.jpg . # move jpg files back into home dir


curl 'e1.ru' # see URL. show html

curl -L 'e1.ru' # see html . Git Bash

curl 'e1.ru'

# git bash save a webpage to a file

# outpot. filename. 
curl -o e1_file.html -L 'e1.ru' #gitbash. -L is gitbash specific. I think it means follow redirects

# grab the whole webpage
curl -o e1_file.html 'e1.ru'

# save a Ekb image from web
curl -o ekb.jpg  'https://www.aeroflot.ru/media/aflfiles/travelguide/ru/svx/svx-1.jpg'

curl 'www.albertacanada.com'





# download excel file https://open.alberta.ca/opendata/alberta-non-profit-listing
curl -o 'non-profits.xlsx' 'https://open.alberta.ca/dataset/bcc15e72-fe46-4215-8de0-33951662465e/resource/23603511-908c-40d3-ab7f-8b59bb54414a/download/non-profit-name-list-for-open-data-portal.xlsx'

cat 'non-profits.xlsx' # not printing it in a readable format

# download a file and print it to console
curl -o dictionary.txt 'https://tinyurl.com/zeyq9vc'
cat dictionary.txt

less dictionary.txt #only works in git bash. print only one screensful of file


# remove , bypasses trash
rmdir NEW # delete folder 

rm *.xlsx # delete
rm -i *.xlsx # remove but it'll ask you if you're sure. Git bash






# grep . does file contain word

grep shell dictionary.txt # git bash


grep shell dictionary.txt | less # git bash pipe into less command. show less output. q to quit


# git bash
# how many time a word occurs on a webpage. results not quite accurate? i don't understand what exactly it counts
curl -L https://en.wikipedia.org/wiki/Yekaterinburg | grep government | wc -l
# wc is word count. -l counts lines. -w counts words

# https://stackoverflow.com/questions/247234/do-you-know-a-similar-program-for-wc-unix-word-count-command-on-windows
# curl  https://en.wikipedia.org/wiki/Yekaterinburg greb government | netstat -an | find /c "government"
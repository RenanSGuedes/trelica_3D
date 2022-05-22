# importing pandas as pd
import pandas as pd
  
# creating the dataframe
df = pd.DataFrame({"Name": ['Anurag', 'Manjeet', 'Shubham', 
                            'Saurabh', 'Ujjawal'],
                     
                   "Address": ['Patna', 'Delhi', 'Coimbatore',
                               'Greater noida', 'Patna'],
                     
                   "ID": [20123, 20124, 20145, 20146, 20147],
                     
                   "Sell": [140000, 300000, 600000, 200000, 600000]})
  
print("Original DataFrame :")
html = df.to_html()

text_file = open("index.html", "w")
text_file.write(f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
    <h1>Hello, world! :smile:</h1>
    {html}
</body>
</html>
''')

text_file.close()
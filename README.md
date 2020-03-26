# textualjoin
This library allows one to Join two pandas dataframe based on text columns.

### How To Install
```pip install textualjoin```

**You may also need to install:**

```python -m spacy download pt```

```python -m spacy download en```

### Associated Libraries
The following libraries are used in textualjoin:
- pandas
- numpy
- nltk
- spacy
- scikit-learn
- nested_dict
- unidecode

### Examples of Use
#### Example 1: join two simple dataframes
- Right and Left dataframes:
```
left_df = 
        values
0	Camiseta M amarela
1	Calça jeans GG
2	Blusa lã

right_df = 
        values
0	Blusa listrada
1	Camis. M
2	jeans
```
- Using textualjoin:
```
TJ = TextualJoin(in_df=left_df,
                 out_df=right_df,
                 text_key='values',
                 language='portuguese')
df = TJ.fit()

Output:
df =
index	        in                             out
0	(1,)	Camiseta M amarela	       Camis. M
1	(1,)	Calça jeans GG	               jeans
2	(1,)	Blusa lã	               Blusa listrada
```

#### Example 2: join two dataframes with aggregation:
**Noitce**: multiple aggregation keys can be used!

- Right and Left dataframes:
```
left_df =
        values	               id
0	Camiseta M amarela	1
1	Calça jeans GG	        1
2	Blusa lã	        2

right_df =
        values	               id
0	Blusa listrada	       2
1	Camis. M	       1
2	jeans	               1
```

- Using textualjoin:
```
TJ = TextualJoin(in_df=left_df,
                 out_df=right_df,
                 aggregation_keys_arr=['id'],
                 text_key='values',
                 language='portuguese')
           
 df = TJ.fit()
 
 Output:
 
 df = 
 index	        in	                out
0	(1,)	Camiseta M amarela	Camis. M
1	(1,)	Calça jeans GG	        jeans
2	(2,)	Blusa lã	        Blusa listrada
```

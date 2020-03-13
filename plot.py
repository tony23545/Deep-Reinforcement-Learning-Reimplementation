import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# This i s j u s t a dummy f u n c ti o n t o g e n e r a t e some a r b i t r a r y data
d e f g e t d a t a ( ) :
b a se c ond = [ [ 1 8 , 2 0 , 1 9 , 1 8 , 1 3 , 4 , 1 ] ,
[ 2 0 , 1 7 , 1 2 , 9 , 3 , 0 , 0 ] ,
[ 2 0 , 2 0 , 2 0 , 1 2 , 5 , 3 , 0 ] ]
cond1 = [ [ 1 8 , 1 9 , 1 8 , 1 9 , 2 0 , 1 5 , 1 4 ] ,
[ 1 9 , 2 0 , 1 8 , 1 6 , 2 0 , 1 5 , 9 ] ,
[ 1 9 , 2 0 , 2 0 , 2 0 , 1 7 , 1 0 , 0 ] ,
[ 2 0 , 2 0 , 2 0 , 2 0 , 7 , 9 , 1 ] ]
cond2= [ [ 2 0 , 2 0 , 2 0 , 2 0 , 1 9 , 1 7 , 4 ] ,
[ 2 0 , 2 0 , 2 0 , 2 0 , 2 0 , 1 9 , 7 ] ,
[ 1 9 , 2 0 , 2 0 , 1 9 , 1 9 , 1 5 , 2 ] ]
cond3 = [ [ 2 0 , 2 0 , 2 0 , 2 0 , 1 9 , 1 7 , 1 2 ] ,
[ 1 8 , 2 0 , 1 9 , 1 8 , 1 3 , 4 , 1 ] ,
[ 2 0 , 1 9 , 1 8 , 1 7 , 1 3 , 2 , 0 ] ,
[ 1 9 , 1 8 , 2 0 , 2 0 , 1 5 , 6 , 0 ] ]
r e t u r n ba se cond , cond1 , cond2 , cond3
# Load the data .
r e s u l t s = g e t d a t a ( )
f i g = p l t . f i g u r e ( )
# We w i l l pl o t i t e r a t i o n s 0 . . . 6
xdata = np . a r r a y ( [ 0 , 1 , 2 , 3 , 4 , 5 , 6 ] ) / 5 .
# Pl o t each l i n e
# (may want t o automate t h i s p a r t e . g . with a l o o p ) .
sns.tsplot(time=xdata , data=r e s u l t s [ 0 ] , c o l o r =’ r ’ , l i n e s t y l e =’−’)
sns . t s p l o t ( time=xdata , data=r e s u l t s [ 1 ] , c o l o r =’g ’ , l i n e s t y l e =’−−’)
sns . t s p l o t ( time=xdata , data=r e s u l t s [ 2 ] , c o l o r =’b ’ , l i n e s t y l e = ’: ’ )
sns . t s p l o t ( time=xdata , data=r e s u l t s [ 3 ] , c o l o r =’k ’ , l i n e s t y l e = ’ −. ’)
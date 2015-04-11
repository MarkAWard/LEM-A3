rm(list=ls())

library(foreign)

setwd("~/Dropbox/DeepLearning/A3")


db=read.csv("data/train.csv",header = F)

shuffle=sample(dim(db)[1],dim(db)[1],replace = F)
db=db[shuffle,]


x=((dim(db)[1])*.8)
train=db[1:x,]

test=db[x:nrow(db),]

write.table(train,file="data/TR_set.csv",row.names = F,sep=",",col.names = F)
write.table(test,file="data/TS_set.csv",row.names = FALSE,sep=",",col.names = FALSE)



# func <- function(x) length(unlist(strsplit(as.character(x)," ")))

#db$length=sapply(X = db[,2],FUN = func,simplify = T,)

# cl <- makeCluster(10)

# x=parLapply(cl, db[,2], func)


rm(list=ls())
load("data/chunky.rda")
load("data/db.rda")

db$length=unlist(x)

par( mfrow = c( 3, 2 ) )
for (i in 1:5)
{
  hist(db$length[(db$V1==i) &  (db$length<600) ],breaks = 25,main=i) 
}


rm(list=ls())
library(foreign)
setwd("~/Dropbox/DeepLearning/A3/data/")
db=read.csv("TS_set.csv",header = F)

x=((dim(db)[1]) %/% 2)

validation=db[1:x,]
test=db[(x+1):nrow(db),]

write.table(validation,file="validation.csv",row.names = FALSE,sep=",",col.names = FALSE)
write.table(test,file="test.csv",row.names = FALSE,sep=",",col.names = FALSE)





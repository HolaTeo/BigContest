rm(list=ls())
#setwd('C:/Users/Industrial Stat Lab/Desktop')
#setwd('C:/Users/lswsi/OneDrive/바탕 화면/대학교/20-여름방학/공모전/2020빅콘테스트 문제데이터(데이터분석분야-챔피언리그)/01_제공데이터')
#setwd('C:/Users/lswsi/Desktop/2020빅콘테스트 문제데이터(데이터분석분야-챔피언리그)/01_제공데이터')
setwd('C:\\Users\\62190\\Documents\\BigContest\\datas')
library(readxl)
library(randomForest)
library(ggplot2)
library(GGally)
library(caret)
library(e1071)
library(gbm)
library(dplyr)
library(xgboost)
library(tidytext)
library(tm)
library(text2vec)
library(wordcloud)
library(SnowballC)
library(stringr)
library(data.table)
library(mltools)
library(FactoMineR)
library(factoextra)
library(lightgbm)
library(Matrix)

# install.packages('randomForest')
# install.packages('GGally')
# install.packages('e1071')
# install.packages('caret')
# install.packages('gbm')
# install.packages('xgboost')
# install.packages('tidytext')
# install.packages('text2vec')
# install.packages('tm')
# install.packages('wordcloud')
# install.packages('SnowballC')
# install.packages('stringr')
# install.packages('data.table')
# install.packages('mltools')
# install.packages('FactoMineR')
# install.packages('factoextra')
# install.packages('tidyverse')
# install.packages('mlr')
# install.packages('Metrics')
# install.packages('Matrix')

# write down at the terminal tab, 
# previously install

# 1. CMake (https://cmake.org/download/)
# 
# 2. git (https://git-scm.com/download/win)
# 
# 3. Rtools (https://cran.r-project.org/bin/windows/Rtools) 
# 
# ( 설치 과정중에, 환경변수를 추가하는 옵션 체크 해줄것)
# 
# 4. Visual Studio (https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=Community&rel=15) 
# 
# (설치 후, 재부팅 필수)


git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
Rscript build_r.R


PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0rc1/lightgbm_3.0.0-1-r-cran.tar.gz"
remotes::install_url(PKG_URL)


PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0rc1/lightgbm-3.0.0-1-r40-windows.zip"
local_file <- paste0("lightgbm.", tools::file_ext(PKG_URL))

download.file(
  url = PKG_URL
  , destfile = local_file
)
install.packages(
  pkgs = local_file
  , type = "binary"
  , repos = NULL
)


# install.packages('devtools')
# library(devtools)
# 
devtools::install_github("Laurae2/lgbdl")
# 
options(devtools.install.args = "--no-multiarch")

devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
# 


# running code to verify successful installation of package

library(lightgbm)
data(agaricus.train, package='lightgbm')
train <- agaricus.train
train %>% head
dtrain <- lgb.Dataset(train$data, label=train$label)
dtrain
params <- list(objective="regression", metric="l2")
model <- lgb.cv(params, dtrain, 10, nfold=5, min_data=1, learning_rate=1, early_stopping_rounds=10)
model


#custom MAPE function for xgboost use feval

MAPE <- function(preds,dtrain){
  labels <- getinfo(dtrain, 'label')
  my_mape <- sum(abs((as.numeric(labels)-as.numeric(preds))/(as.numeric(preds))))*100
  my_mape <- my_mape/length(as.numeric(preds))
  return(list(metric='mape',value=my_mape))
}


# dataset reading

d1 <- read.csv('나만의데이터.csv')
str(d1)
d1 <- as.matrix(d1)
d1%>%head
d1 %>% dim
d1 <- na.omit(d1)

day <- (d1[,2])
day <- as.factor(day)
day

date <- d1[,1]

month <- as.factor(d1[,3])

time <- as.factor(d1[,4])

con_time <- as.numeric(d1[,5])

exposure <- as.numeric(d1[,9])

brand <- as.factor(d1[,10])

code_name <- as.factor(d1[,11])

merch_name <- tolower(d1[,12])

category <- (d1[,13])
category %>% unique
category[category=='의류'] <- 1
category[category=='속옷'] <- 2
category[category=='주방'] <- 3
category[category=='농수축'] <- 4
category[category=='이미용'] <- 5
category[category=='가전'] <- 6
category[category=='생활용품'] <- 7
category[category=='건강기능'] <- 8
category[category=='잡화'] <- 9
category[category=='가구'] <- 10
category[category=='침구'] <- 11
category <- factor(category)
category

price <- as.numeric(d1[,20])

total_revenue <- as.numeric(d1[,23])

seemean <- as.numeric(d1[,21])
min(seemean[seemean!=0])
seemean[seemean==0] <- 0.00006

precipitation <- as.numeric(d1[,15])

mean_temp <- as.numeric(d1[,14])

cold_sc <- as.numeric(d1[,16])

flu_sc <- as.numeric(d1[,17])

pneumonia_sc <- as.numeric(d1[,18])

coronavirus_sc <- as.numeric(d1[,19])

data0 <- data.frame(day,date,month,time,con_time,exposure,brand,code_name,merch_name,category,price,
                    total_revenue,seemean,precipitation,mean_temp,cold_sc,flu_sc,pneumonia_sc,coronavirus_sc)
sum(is.na(data0))


#Giving seq to data
data0 <-  arrange(data0,code_name)
data0 <-  arrange(data0,brand)
View(data0)

sell_sequence <- rep(NA, length(data0$code_name))
sell_sequence[1] <- 1
for(i in 1:length(data0$code_name)){
  ifelse((data0$date[i]==data0$date[i+1]& data0$code_name[i]==data0$code_name[i+1]
          & data0$day[i]==data0$day[i+1])
         ,sell_sequence[i+1] <- sell_sequence[i]+1, sell_sequence[i+1] <- 1 )
}
sell_sequence
sum(is.na(sell_sequence))
sell_sequence[sell_sequence==7] <- 1
sell_sequence[sell_sequence==8] <- 2
sell_sequence[sell_sequence==9] <- 3
sell_sequence[sell_sequence==10] <- 4
sell_sequence[sell_sequence==11] <- 5
sell_sequence[sell_sequence==12] <- 6

sell_sequence <- factor(sell_sequence,order=T,levels=c(1,2,3,4,5,6))

data00 <- data.frame(data0,sell_sequence)
head(data00)
str(data00)

data_seq1 <- data00[data00$sell_sequence==1,]
data_seq2 <- data00[data00$sell_sequence==2,]
data_seq3 <- data00[data00$sell_sequence==3,]
data_seq4 <- data00[data00$sell_sequence==4,]
data_seq5 <- data00[data00$sell_sequence==5,]
data_seq6 <- data00[data00$sell_sequence==6,]

data_seq_mean <- data.frame(mean(data_seq1$total_revenue),mean(data_seq2$total_revenue),mean(data_seq3$total_revenue),
                            mean(data_seq4$total_revenue),mean(data_seq5$total_revenue),mean(data_seq6$total_revenue))
data_seq_var <- data.frame(var(data_seq1$total_revenue),var(data_seq2$total_revenue),var(data_seq3$total_revenue),
                           var(data_seq4$total_revenue),var(data_seq5$total_revenue),var(data_seq6$total_revenue))
order(data_seq_mean)
data_seq_mean[order(data_seq_mean)]

rank_seq_mean <- rep(0,length(data00$total_revenue))
for(i in 1:length(data00$total_revenue)){
  if(data00$sell_sequence[i]==1){rank_seq_mean[i] <- 1}
  else(if(data00$sell_sequence[i]==5){rank_seq_mean[i] <- 2}
       else(if(data00$sell_sequence[i]==2){rank_seq_mean[i] <- 3}
            else(if(data00$sell_sequence[i]==4){rank_seq_mean[i] <- 4}
                 else(if(data00$sell_sequence[i]==6){rank_seq_mean[i] <- 5}
                      else(if(data00$sell_sequence[i]==3){rank_seq_mean[i] <- 6})))))
}
unique(rank_seq_mean)
str(rank_seq_mean)

order(data_seq_var)
data_seq_var[order(data_seq_var)]

rank_seq_var <- rep(0,length(data00$total_revenue))
for(i in 1:length(data00$total_revenue)){
  if(data00$sell_sequence[i]==1){rank_seq_var[i] <- 1}
  else(if(data00$sell_sequence[i]==5){rank_seq_var[i] <- 2}
       else(if(data00$sell_sequence[i]==2){rank_seq_var[i] <- 3}
            else(if(data00$sell_sequence[i]==6){rank_seq_var[i] <- 4}
                 else(if(data00$sell_sequence[i]==3){rank_seq_var[i] <- 5}
                      else(if(data00$sell_sequence[i]==4){rank_seq_var[i] <- 6})))))
}
unique(rank_seq_var)
str(rank_seq_var)

#giving rank to brand name
data_merch_name <- read_xlsx('seungwonrawdata.xlsx')
data_merch_name %>% head
data_merch_name <- as.matrix(data_merch_name)
brand_name <- data_merch_name[,10]
brand_name %>% head


corpus_top_name <- Corpus(VectorSource(brand_name),
                          readerControl=list(language='kor'))
corpus_top_name <- tm_map(corpus_top_name,content_transformer(tolower))
corpus_top_name <- tm_map(corpus_top_name,removePunctuation)

text_top_name <- TermDocumentMatrix(corpus_top_name)
dtm_top_name <- as.matrix(text_top_name)
dtm_sum_top_merch_name <- sort(rowSums(dtm_top_name),decreasing=F)
dtm_df_top_merch_name <- data.frame(word=names(dtm_sum_top_merch_name),
                                    freq=dtm_sum_top_merch_name)
dtm_df_top_merch_name %>% head(10)
#wordcloud(words=dtm_df_top_merch_name$word, freq=dtm_df_top_merch_name$freq,
#min.freq=100,max.words=100,random.order = F,rot.per=0.15,
#colors=brewer.pal(5,'Dark2'))
top_brand_name <- rownames(dtm_df_top_merch_name)

rank_brand <- rep(1, length(data00$merch_name))
rank_brand[grep('삼성',data00$merch_name)]

for(i in 1:length(top_brand_name)){
  rank_brand[grep(top_brand_name[i],data00$merch_name)] <- i
}
length(unique(rank_brand))

data00 <- data.frame(data00,rank_seq_mean,rank_seq_var,rank_brand)
#data00$rank_brand <- factor(data00$rank_brand,order=T)
data00$rank_brand <- as.numeric(data00$rank_brand)
data00$temp_diff <- data00$top_temp-data00$bottom_temp

#XG boost
# 
# 
set.seed(123)

#data00 <- data00[data00$total_revenue!=50000,]

head(data00)
new_data00 <- select(data00,total_revenue,day, month, time, con_time, category, price, 
                     #seemean, seevar, mean_temp, top_temp, bottom_temp, rank_seq_var
                     precipitation, temp_diff, mean_temp,
                     sell_sequence, rank_seq_mean,rank_brand)
head(new_data00)
str(new_data00)

# ggplot(data=new_data00, aes(x=precipitation, y=total_revenue))+
#   geom_point(size=2)
# 
# unique(new_data00$precipitation)#0, 9.4, 28.9, 56.5
# 
# ggplot(data=new_data00[new_data00$precipitation>=9.4,], aes(x=precipitation, y=total_revenue))+
#   geom_point(size=2)
# 
# ggplot(data=new_data00, aes(x=mean_temp, y=total_revenue))+
#   geom_point(size=2)
# 
# ggplot(data=new_data00, aes(x=top_temp, y=total_revenue))+
#   geom_point(size=2)
# 
# ggplot(data=new_data00, aes(x=bottom_temp, y=total_revenue))+
#   geom_point(size=2)
# 
# ggplot(data=new_data00, aes(x=temp_diff, y=total_revenue))+
#   geom_point(size=2)


new_data00$sell_sequence <- as.numeric(new_data00$sell_sequence)
new_data00$rank_seq_mean <- as.numeric(new_data00$rank_seq_mean)
#new_data00$rank_seq_var <- as.numeric(new_data00$rank_seq_var)


# category_precipitation <- rep(NA, length(new_data00$precipitation))
# for( i in 1: length(new_data00$precipitation)){
#   if(new_data00$precipitation[i]>=0&new_data00$precipitation[i]<9.4){
#     category_precipitation[i] <- 4
#   }else(if(new_data00$precipitation[i]>=9.4&new_data00$precipitation[i]<28.9){
#     category_precipitation[i] <- 3
#   }else(if(new_data00$precipitation[i]>=28.9&new_data00$precipitation[i]<56.5){
#     category_precipitation[i] <- 2
#   }else(if(new_data00$precipitation[i]>=56.5){
#     category_precipitation[i] <- 1
#   })))
# }
# sum(is.na(category_precipitation))
# c_precipitation <- as.numeric(category_precipitation)
# c_precipitation
# 
# new_data00 %>% dim
# new_data00 <- data.frame(new_data00, c_precipitation)
# new_data00 %>% head


# new_data00 %>% head
# View(new_data00 %>% filter(total_revenue>=100000000))
# new_data00 %>% filter(total_revenue>=100000000) %>% select(month) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(category) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(time) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(price) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(sell_sequence) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(rank_brand) %>% unlist() %>% as.numeric() %>% hist()
# new_data00 %>% filter(total_revenue>=100000000) %>% select(rank_seq_mean) %>% unlist() %>% as.numeric() %>% hist()
# 
# 
# 
# 
# 
# new_data00 %>% filter(total_revenue==50000) %>% dim()
# plot(sort(new_data00$total_revenue[new_data00$total_revenue>90000000],decreasing=F))
# length(new_data00$total_revenue[new_data00$total_revenue>90000000])
# new_data00$total_revenue[order(new_data00$total_revenue,decreasing=T)]
# new_data00$total_revenue %>% quantile(c(0.996,0.997,0.998,0.999))
#new_data001 <- new_data00[new_data00$category==1,]
#new_data001 %>% head

# ggplot(data=new_data00, aes(x=rank_brand,y=total_revenue))+
#   geom_point(size=2)+
#   geom_smooth(method='lm')
#   
# ggplot(data=new_data00, aes(x=seemean,y=total_revenue))+
#   geom_point(size=2)+
#   geom_smooth(method='lm')
# 
# ggplot(data=new_data00, aes(x=seemax,y=total_revenue))+
#   geom_point(size=2)+
#   geom_smooth(method='lm')

# ggplot(data=new_data00, aes(x=seevar,y=total_revenue))+
#   geom_point(size=2)+
#   geom_smooth(method='lm')


#FAMD for data (PCA approach both for categorical and numerical)
# res.famd <- FAMD(new_data00,ncp=10,graph=F)
# eig.val <- get_eigenvalue(res.famd)
# eig.val
# 
# vari <- get_famd_var(res.famd)
# vari$contrib
# 
# fviz_contrib(res.famd,'var',repel=T,col.var='contrib',axes=1)

#XG boost
index <- sample(1:nrow(new_data00),size=round(0.75*nrow(new_data00)),replace=F)
trs1 <- new_data00[index,]
trs1 %>% head
tts1 <- new_data00[-index,]
# 
# trs_labels1 <- as.numeric(trs1$category_50000)-1
# str(trs_labels1)
# tts_labels1 <- as.numeric(tts1$category_50000)-1
# new_trs1 <- model.matrix(~.,trs1[-1])
# head(new_trs1)
# new_tts1 <- model.matrix(~.,tts1[-1])
# head(new_tts1)
# 
# 
# xg_train1 <- xgb.DMatrix(data=new_trs1,label=trs_labels1)
# xg_test1 <- xgb.DMatrix(data=new_tts1,label=tts_labels1)    
# str(xg_train1)
# 
# 
# def_param1 <- list(booster='gbtree',objective='binary:logistic',eta=0.3,gamma=0,max_depth=6,
#                    min_child_weight=1,subsample=1,colsample_bytree=1)
# 
# xgbcv1 <- xgb.cv(params=def_param1,data=xg_train1,nrounds=100,nfold=5,
#                  showsd=T,stratified = T,print_every_n = 5,
#                  early_stopping_rounds = 20,maximize = F)
# 
# min(xgbcv1$test.error.mean)
# 
# xgb1 <- xgb.train(params=def_param1,data=xg_train1,nrounds=65,
#                   watchlist = list(val=xg_test1,train=xg_train1),
#                   print_every_n =10,early_stopping_rounds = 20,
#                   maximize=F,eval_matrix="error")
# 
# xgbpred1 <- predict(xgb1,xg_test1)
# xgbpred1 <- ifelse(xgbpred1>0.5,1,0)
# 
# xgbpred1
# tts_labels1
# confusionMatrix(as.factor(xgbpred1),as.factor(tts_labels1))
# 
# imp_mat1 <- xgb.importance(feature_names = colnames(new_trs1),model=xgb1)
# imp_mat1
# xgb.plot.importance(importance_matrix = imp_mat1)


#iteration 1
trs_labels1 <- (trs1$total_revenue)
str(trs_labels1)
tts_labels1 <- (tts1$total_revenue)
new_trs1 <- model.matrix(~.,trs1[-1])
head(new_trs1) %>% dim
new_tts1 <- model.matrix(~.,tts1[-1])
head(new_tts1)


xg_train1 <- xgb.DMatrix(data=new_trs1,label=trs_labels1)
xg_test1 <- xgb.DMatrix(data=new_tts1,label=tts_labels1)    

def_param1 <- list(booster='gbtree',objective='reg:squarederror',eta=0.1,gamma=0,max_depth=6,
                   min_child_weight=1,subsample=1,colsample_bytree=1)

xgbcv1 <- xgb.cv(params=def_param1,data=xg_train1,nrounds=5000,nfold=5,
                 showsd=T,stratified = T,print_every_n = 1,
                 early_stopping_rounds = 1,maximize = F,eval_metric=MAPE)

which.min(xgbcv1$evaluation_log$test_mape_mean)

xgb1 <- xgb.train(params=def_param1,data=xg_train1,nrounds=57,
                  watchlist = list(train=xg_train1,test=xg_test1),
                  print_every_n = 1,early_stopping_rounds = 20,
                  maximize= F, eval_metric= 'mae')

which.min(xgb1$evaluation_log$test_mae)

xgbpred1 <- predict(xgb1,xg_test1)
xgbpred1
chk_xgb1 <- data.frame(original=tts1$total_revenue,prediction=(xgbpred1))
#chk_xgb1[chk_xgb1$prediction<0,]


# chk <- data.frame(original=tts1,prediction=xgbpred1)
# chk[chk$prediction<0,] %>% head
# chk[chk$prediction<0,] %>% dim
# sum(chk[chk$prediction<0,]$original.sell_sequence==1)
# chk[chk$prediction<0,] %>% filter(original.sell_sequence==1)


# min(chk_xgb1$prediction[chk_xgb1$prediction>=0])
# mean(chk_xgb1$prediction[chk_xgb1$prediction>=0])
# median(chk_xgb1$prediction[chk_xgb1$prediction>=0])
# 
#chk_xgb1$prediction[chk_xgb1$prediction<0] <- 2500000
#chk_xgb1$prediction[chk_xgb1$prediction<0] <- (-1)*(chk_xgb1$prediction[chk_xgb1$prediction<0])
# (chk_xgb1$prediction[chk_xgb1$prediction<0])%>% max

chk_xgb1
sum(abs((chk_xgb1$prediction-chk_xgb1$original)/(chk_xgb1$prediction)))*100/length(chk_xgb1$prediction)

imp_mat1 <- xgb.importance(feature_names = colnames(new_trs1),model=xgb1)
imp_mat1
xgb.plot.importance(importance_matrix = imp_mat1)


# find_non_zero <- function(trs1,tts1){
#   x <- vector(mode='list',length=1000)
#   y <- vector(mode='list',length=1000)
#   w <- vector(mode='list',length=1000)
#   z <- rep(NA,1000)
#   x[[1]] <- tts1
#   new_trs1 <- model.matrix(~.,trs1[-1])
#   xg_train1 <- xgb.DMatrix(data=new_trs1,label=trs1$total_revenue)
#   def_param1 <- list(booster='gbtree',objective='reg:squarederror',eta=0.1,gamma=0,max_depth=8,
#                      min_child_weight=1,subsample=1,colsample_bytree=1)
# 
#   new_tts1 <- model.matrix(~.,x[[1]][-1])
#   xg_test1 <- xgb.DMatrix(data=new_tts1,label=x[[1]]$total_revenue)
#   y[[1]] <- xgb.cv(params=def_param1,data=xg_train1,nrounds=2000,nfold=5,
#                    showsd=T,stratified = T,print_every_n = 50,
#                    early_stopping_rounds = 20,maximize = F)
# 
#   xgb1 <- xgb.train(params=def_param1,data=xg_train1,nrounds=which.min(y[[1]]$evaluation_log$test_rmse_mean),
#                     watchlist = list(val=xg_test1,train=xg_train1),
#                     print_every_n = 50,early_stopping_rounds = 20,
#                     maximize=F,eval_matrix="error")
# 
#   xgbpred1 <- predict(xgb1,xg_test1)
#   w[[1]] <- data.frame(original=x[[1]]$total_revenue,prediction=xgbpred1)
# 
#   z[1] <- sum(abs((w[[1]]$prediction-w[[1]]$original)/(w[[1]]$prediction)))*100/length(w[[1]]$prediction)
# 
#   #start of new iteration
#   for(i in 1:1000){
#     if(length(w[[i]]$prediction[w[[i]]$prediction<0])!=0){
#       print('not yet')
#       xgb_train2 <- data.frame(x[[1]],prediction=w[[1]]$prediction)
# 
#       x[[i+1]] <- xgb_train2[xgb_train2$prediction<0,][,-13]
# 
#       tts_labels2 <- x[[i+1]]$total_revenue
#       new_tts2 <- model.matrix(~.,x[[i+1]][-1])
# 
#       xg_test2 <- xgb.DMatrix(data=new_tts2,label=tts_labels2)
# 
#       y[[i+1]] <- xgb.cv(params=def_param1,data=xg_train1,nrounds=2000,nfold=5,
#                          showsd=T,stratified = T,print_every_n = 150,
#                          early_stopping_rounds = 20,maximize = F)
# 
#       xgb2 <- xgb.train(params=def_param1,data=xg_train1,nrounds=which.min(y[[i+1]]$evaluation_log$test_rmse_mean),
#                         watchlist = list(val=xg_test2,train=xg_train1),
#                         print_every_n = 150,early_stopping_rounds = 20,
#                         maximize=F,eval_matrix="error")
# 
#       xgbpred2 <- predict(xgb2,xg_test2)
# 
#       w[[i+1]] <- data.frame(original=x[[i+1]]$total_revenue,prediction=xgbpred2)
#       w[[1]][w[[1]]$prediction<0,] <- w[[i+1]]
# 
#       z[i+1] <- sum(abs((w[[1]]$prediction-w[[1]]$original)/(w[[1]]$prediction)))*100/length(w[[1]]$prediction)
#     }
#     else(if(length(w[[i]]$prediction[w[[i]]$prediction<0])==0){
#       break})
#   }
#   return(list(prediction=z,chk=w[[1]]))
# 
# }
# 
# a <- find_non_zero(trs1,tts1)
# a
# 
# sum(a$chk<0)
# min(a$prediction)


#hyperparameter tuning

# Create empty lists
# lowest_error_list = list()
# parameters_list = list()

# Create 10,000 rows with random hyperparameters
# set.seed(123)
# for (iter in 1:10000){
#   param <- list(booster = "gbtree",
#                 objective = "reg:squarederror",
#                 max_depth = sample(3:10, 1),
#                 eta = runif(1, 0.01, 0.3),
#                 subsample = runif(1, 0.5, 0.8),
#                 colsample_bytree = runif(1, 0.5, 0.9),
#                 min_child_weight = sample(0:10, 1)
#   )
#   parameters <- as.data.frame(param)
#   parameters_list[[iter]] <- parameters
# }
# 
# parameters_list
# # Create object that contains all randomly created hyperparameters
# parameters_df = do.call(rbind, parameters_list)
# nrow(parameters_df[1,])
# parameters_df %>% head
# # x<-list(c(1,2,3),c(4,5,6))
# # x
# # lapply(x,sum)
# # lapply(x,rbind)
# # do.call(sum,x)
# # do.call(rbind,x)
# 
# # Use randomly created parameters to create 10,000 XGBoost-models
# for (row in 1:nrow(parameters_df)) {
#   set.seed(123)
#   best_iteration <- matrix(NA,nrow = 10, ncol=2)
#   for(j in 1:10){
#     xgbcv1 <- xgb.cv(data=xg_train1,nrounds=5000,nfold=5,
#                      max_depth = parameters_df$max_depth[row],
#                      eta = parameters_df$eta[row],
#                      subsample = parameters_df$subsample[row],
#                      colsample_bytree = parameters_df$colsample_bytree[row],
#                      min_child_weight = parameters_df$min_child_weight[row],
#                      showsd=T,stratified = T,print_every_n = 1,
#                      early_stopping_rounds = 1,maximize = F,eval_metric=MAPE)
#     best_iteration[j,] <- c(which.min(xgbcv1$evaluation_log$test_mape_mean),min(xgbcv1$evaluation_log$test_mape_mean))
#   }
#   
#   xgb1 <- xgb.train(data=xg_train1,
#                     booster = "gbtree",
#                     objective = "reg:squarederror",
#                     max_depth = parameters_df$max_depth[row],
#                     eta = parameters_df$eta[row],
#                     subsample = parameters_df$subsample[row],
#                     colsample_bytree = parameters_df$colsample_bytree[row],
#                     min_child_weight = parameters_df$min_child_weight[row],
#                     nrounds= best_iteration[which.min(best_iteration[,2])],
#                     eval_metric = "mae",
#                     early_stopping_rounds= 20,
#                     print_every_n = 150,
#                     watchlist = list(train=xg_test1,test=xg_train1))
#   xgbpred <- predict(xgb1,xg_test1)
#   chk_xgb <- data.frame(original=tts1$total_revenue,prediction=xgbpred)
#   lowest_error <-  sum(abs((chk_xgb$prediction-chk_xgb$original)/(chk_xgb$prediction)))*100/length(chk_xgb$prediction)
#   lowest_error_list[row] <- lowest_error
# }
# 
# # Create object that contains all accuracy's
# lowest_error_df <-  do.call(rbind, lowest_error_list)
# lowest_error_df
# 
# # Bind columns of accuracy values and random hyperparameter values
# randomsearch <-  cbind(lowest_error_df, parameters_df)
# randomsearch

# light gbm

MAPE2 <- function(preds,dtrain){
  labels <- getinfo(dtrain, 'label')
  my_mape <- sum(abs((as.numeric(labels)-as.numeric(preds))/(as.numeric(preds))))*100
  my_mape <- my_mape/length(as.numeric(preds))
  return(list(name='mape',value=my_mape,higher_better=F))
}

new_trs1 %>% head
new_trs1 %>% str
as.data.frame(new_trs1)
lg_trainm <- sparse.model.matrix(total_revenue~., data=trs1)
lg_train_label <- trs1$total_revenue
lg_testm <- sparse.model.matrix(total_revenue~., data=tts1)
lg_test_label <- tts1$total_revenue

lg_train <- lgb.Dataset(data=as.matrix(lg_trainm),label=lg_train_label)
lg_train
lg_test <- lgb.Dataset(data=as.matrix(lg_testm),label=lg_test_label) 
lg_test
getinfo(lg_train,'label')

def_param2 <- list(boosting ='gbdt',objective='regression', num_leaves= 31, max_depth= -1,
                   feature_fraction=0.7, bagging_fraction=0.7,
                   bagging_freq=5, learning_rate=0.1, num_threads=2)

lgbcv1 <- lgb.cv(params=def_param2,data=lg_train, nrounds=5000,
                 early_stopping_rounds = 20, eval_freq = 150,
                 nfold=5, showsd=5, stratified = T,verbose=1,eval=MAPE2)

lgbcv1$best_iter

lgb1 <- lgb.train(params=def_param2,objective='regression',data=lg_train,
                  nrounds=lgbcv1$best_iter,
                  eval_freq=150)

lgbpred1 <- predict(lgb1,new_tts1)
lgbpred1
chk_lgb1 <- data.frame(original=tts1$total_revenue,prediction=(lgbpred1))
chk_lgb1
chk_lgb1[chk_lgb1$prediction<0,]
chk_lgb1
sum(abs((chk_lgb1$prediction-chk_lgb1$original)/(chk_lgb1$prediction)))*100/length(chk_lgb1$prediction)

imp_mat2 <- lgb.importance(model=lgb1,percentage=T)
imp_mat2
lgb.plot.importance(imp_mat2,measure="Gain", top_n=60)

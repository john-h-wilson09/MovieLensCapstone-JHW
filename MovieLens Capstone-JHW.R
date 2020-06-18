library(tidyverse)
library(dslabs)
library(broom)
library(lubridate)
library(caret)
library(boot)
library(stringr)
library(purrr)
library(knitr)
library(tinytex)
options(digits = 3)    
library(MASS)
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx <- edx %>% mutate(genre = fct_lump(genres, n=50)) #reduce genres to 50 categories

model_index <- createDataPartition(edx$rating, times = 1, p=0.5, list=FALSE)
train_edx <- edx[-model_index,]
test_edx <- edx[model_index,]


# Average Prediction
avg <- rep(mean(train_edx$rating), times=nrow(train_edx))
avg_rmse <- RMSE(avg,test_edx$rating)


# Effects Models
mu <- mean(train_edx$rating)
user_eff <- train_edx %>% group_by(userId) %>% summarize(ue_i = mean(rating-mu))
movie_eff <- train_edx %>% left_join(user_eff, by='userId') %>%
    group_by(movieId) %>%  summarize(me_i = mean(rating-mu-ue_i))
eff_rating <- test_edx %>% left_join(movie_eff, by= "movieId") %>%
    left_join(user_eff, by='userId') %>% mutate(pred = mu + me_i + ue_i) %>% 
    filter(!is.na(pred)) %>% pull(pred) #disregarded all NA ratings
movie_user_eff <- RMSE(eff_rating, test_edx$rating)

genre_eff <- train_edx %>% left_join(user_eff, by='userId') %>% left_join(movie_eff, by="movieId") %>%
  group_by(genre) %>% summarize(ge_i = mean(rating-mu-ue_i-me_i))
eff2_rating <- test_edx %>% left_join(genre_eff, by= "genre") %>% left_join(movie_eff, by= "movieId") %>%
  left_join(user_eff, by='userId') %>% mutate(pred = mu + me_i + ue_i + ge_i) %>% 
  filter(!is.na(pred)) %>% pull(pred) #disregarded all NA ratings
movie_use_genre_eff <- RMSE(eff2_rating, test_edx$rating)


#Regularized Effects Model
lambda <- seq(5,50,5)
pred_rmse <- map(lambda, function(l){
  user_ave <- train_edx %>% group_by(userId) %>% summarize(u_i = sum(rating-mu)/(n()+l))
  movie_ave <- train_edx %>% left_join(user_ave, by='userId') %>%
    group_by(movieId) %>%  summarize(m_i = sum(rating-mu-u_i)/(n()+l))
  
  pred_rating <- test_edx %>% left_join(movie_ave, by= "movieId") %>%
    left_join(user_ave, by='userId') %>% mutate(pred = mu + m_i + u_i) %>% 
    filter(!is.na(pred)) %>% pull(pred) #disregarded all NA ratings
  
  RMSE(pred_rating, test_edx$rating)
  })
lambda[which.min(pred_rmse)]
reg_movie_user_eff <- pred_rmse[[which.min(pred_rmse)]]
regular_effect <- plot(lambda,pred_rmse)


# Regression Models - sampled test and train sets as data was too large for regressions
set.seed(1,sample.kind = "Rounding")
reg_train <- train_edx[sample(nrow(train_edx), 5000,)] 
reg_train <- reg_train[,-5:-6]
reg_test <- test_edx[sample(nrow(test_edx), 5000,)]

knn_fit <- train(rating~., method= "knn", data=reg_train)
knn_preds <- predict(knn_fit,reg_test)
knn_mod <- RMSE(knn_preds, reg_test$rating)

loe_fit <- loess(rating~userId+movieId,reg_train)
loe_preds <- predict(loe_fit,reg_test) 
loe_preds <- loe_preds[!is.na(loe_preds)] #removing NA values
loe_preds[4998:5000] <- loe_preds[996:998] #copied random rows to make table same length as others for ensemble
loe_mod <- RMSE(loe_preds, reg_test$rating)

glm_fit <- train(rating~., method = "glm", data=reg_train)
glm_preds <- predict(glm_fit,reg_test)
glm_mod <- RMSE(glm_preds, reg_test$rating)

ensemble <- data.frame(knn_preds,loe_preds,glm_preds)
ensemble_pred <- rowMeans(ensemble)
ensemble_mod <- RMSE(ensemble_pred, reg_test$rating)


Results <- data.frame(Models=c("Avg Pred","Movie+User Eff","Movie+User+Genre Eff",
                               "Reg Movie+User Eff","Knn", "Loess", "GLM","Ensemble"), 
                      Mod_RMSE=c(avg_rmse,movie_user_eff,movie_use_genre_eff, 
                             reg_movie_user_eff,knn_mod,loe_mod,glm_mod,ensemble_mod)) %>%
          arrange(desc(Mod_RMSE))
Results

# Validation Test - selected GLM
validation <- validation %>% mutate(genre= fct_lump(genres, n=50))
val_preds <- predict(glm_fit, validation)
val_rmse <- data.frame(Model = "Validation RMSE", Val_RMSE=RMSE(val_preds,validation$rating))

knit()

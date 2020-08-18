##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
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

# See structure of dataset
str(edx)

# Check for NA values
sum(is.na(edx))

# Unique items in each variable
unique_variables <- data_frame(Variable=c("Movies","Users","Genres"),
                               Count=c(length(unique(edx$movieId)),length(unique(edx$userId)),
                                       length(unique(edx$genres))))

# See how titles, genres and user vary in count
edx %>% count(movieId)%>%ggplot(aes(n)) + 
  geom_histogram(color = "black")+ scale_x_log10() + labs(title="Movies", x="", y="Count")
edx %>% count(genres)%>%ggplot(aes(n)) + 
  geom_histogram(color = "black")+ scale_x_log10() + labs(title="Genres", x="", y="Count")
edx %>% count(userId) %>% ggplot(aes(n)) + 
  geom_histogram(color = "black")+ scale_x_log10() + labs(title="Users", x="", y="Count")

# See how ratings are spread across years
edx %>% mutate(timestamp = as_datetime(timestamp)) %>% mutate(year = year(timestamp)) %>%
  group_by(year) %>% summarize(avg_rating=mean(rating)) %>%
  ggplot(aes(year,avg_rating)) + geom_smooth()

# See how ratings are spread across time of day
edx %>% mutate(timestamp = as_datetime(timestamp)) %>% mutate(hour = hour(timestamp)) %>%
  group_by(hour) %>% summarize(avg_rating=mean(rating)) %>%
  ggplot(aes(hour,avg_rating)) + geom_smooth()

# See how rating counts are spread across time of day
edx %>% mutate(timestamp = as_datetime(timestamp)) %>% mutate(hour = hour(timestamp)) %>%
  group_by(hour) %>% summarize(Count=n()) %>%
  ggplot(aes(hour,Count)) + geom_smooth() + scale_y_log10()

# See how ratings changed based on day of the week
edx %>% mutate(timestamp = as_datetime(timestamp)) %>% mutate(days = wday(timestamp)) %>%
  group_by(days) %>% summarize(avg_rating=mean(rating)) %>%
  ggplot(aes(days,avg_rating)) + geom_smooth() + scale_y_log10()

# Ratings count by day of the week, Wed is most active
edx %>% mutate(timestamp = as_datetime(timestamp)) %>% mutate(days = wday(timestamp)) %>%
  group_by(days) %>% summarize(Count=n()) %>%
  ggplot(aes(days,Count)) + geom_smooth() + scale_y_log10()

# Top 5 movies with more than 100 ratings
edx %>% group_by(title) %>% summarize(n=n(),avg_rating=mean(rating))%>% filter(n>100) %>% 
  arrange(desc(avg_rating)) %>% head(5)

# Worse 5 movies with more than 100 ratings
edx %>% group_by(title) %>% summarize(n=n(),avg_rating=mean(rating))%>% filter(n>100) %>% 
  arrange(desc(avg_rating)) %>% tail(5)

# Per instructions, the validation data can only be used as a final step so edx needs to be split for training
# Dividing edx into practice and test sets to prepare models prior to validation
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating,times=1,p=0.2,list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]
options(digits = 5)

# Ensuring test set doesn't include any users and movies that do not appear in the training set.
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

# Naive-Average Model
mu <- mean(train_set$rating)
avg_pred <- rep(mu,nrow(test_set))
avg_rmse <- RMSE(test_set$rating,avg_pred)
RMSE_result0 <- data_frame(Model = "Naive-Avg", RMSE = avg_rmse) #create table for results

# Movie Effect Model
movie_avg <- train_set %>% group_by(movieId) %>% summarize(b_m = mean(rating-mu))
movie_pred <- test_set %>% left_join(movie_avg, by='movieId') %>% 
  mutate(pred=mu+b_m) %>% pull(pred)
movie_rmse <- RMSE(movie_pred,test_set$rating)
RMSE_result1 <- bind_rows(RMSE_result0, data_frame(Model="Movie Effect Model", RMSE = movie_rmse ))

# Movie+User Effect Model
user_avg <- train_set %>% left_join(movie_avg, by='movieId') %>% group_by(userId) %>% 
  summarize(b_u = mean(rating-mu-b_m))
user_pred <- test_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  mutate(pred=mu+b_m+b_u) %>% pull(pred)
user_rmse <- RMSE(user_pred,test_set$rating)
RMSE_result1 <- bind_rows(RMSE_result1, data_frame(Model="Movie+User Effect Model", RMSE = user_rmse ))

# Movie+User+Genre Effect Model
genre_avg <- train_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  group_by(genres) %>% summarize(b_g = mean(rating-mu-b_m-b_u))
genre_pred <- test_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(pred=mu+b_m+b_u+b_g) %>% pull(pred)
genre_rmse <- RMSE(genre_pred,test_set$rating)
RMSE_result1 <- bind_rows(RMSE_result1, data_frame(Model="Movie+User+Genre Model", RMSE = genre_rmse ))

# Movie+User+Genre+hr Effect Model
hr_avg <- train_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(hr = hour(as_datetime(timestamp))) %>% group_by(hr) %>% 
  summarize(b_hr = mean(rating-mu-b_m-b_u-b_g))
hr_pred <- test_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(hr = hour(as_datetime(timestamp))) %>% 
  left_join(hr_avg, by='hr') %>% mutate(pred=mu+b_m+b_u+b_g) %>% pull(pred)
hr_rmse <- RMSE(hr_pred,test_set$rating)
RMSE_result2 <- bind_rows(RMSE_result1, data_frame(Model="Movie+User+Genre+hr Model", RMSE = hr_rmse ))

# Movie+User+Genre+day Effect Model
day_avg <- train_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(day = wday(as_datetime(timestamp))) %>% group_by(day) %>% 
  summarize(b_dy = mean(rating-mu-b_m-b_u-b_g))
day_pred <- test_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(day = wday(as_datetime(timestamp))) %>% 
  left_join(day_avg, by='day') %>% mutate(pred=mu+b_m+b_u+b_dy) %>% pull(pred)
day_rmse <- RMSE(day_pred,test_set$rating)
RMSE_result2 <- bind_rows(RMSE_result2, data_frame(Model="Movie+User+Genre+day Model", RMSE = hr_rmse ))

# Regularized Effects Model
lambdas <- seq(3, 15, 0.5)
RMSEreg <- sapply(lambdas, function(l){
    user_ave <- train_set %>% group_by(userId) %>% summarize(u_i = sum(rating-mu)/(n()+l))
    movie_ave <- train_set %>% left_join(user_ave, by='userId') %>%
      group_by(movieId) %>%  summarize(m_i = sum(rating-mu-u_i)/(n()+l))
    
    pred_rating <- test_set %>% left_join(movie_ave, by= "movieId") %>%
      left_join(user_ave, by='userId') %>% mutate(pred = mu + m_i + u_i ) %>% pull(pred) 
  
    RMSE(pred_rating, test_set$rating)
  })
qplot(lambdas,RMSEreg)
lambda <- lambdas[which.min(RMSEreg)] #best lamda was 14
RMSEreg_eff <- RMSEreg[[which.min(RMSEreg)]]
RMSE_result3 <- bind_rows(RMSE_result2, data_frame(Model="Regularization Model", RMSE = RMSEreg_eff ))

# Matrix Factorization Model
# Calculating residuals by removing movie, user and genres effect
residual_set <- train_set %>% left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(resid = rating - mu - b_m - b_u - b_g)

# Setting data and tuning parameters - used R documentation for parameters
reco <- Reco()
train_reco <- data_memory(user_index=residual_set$userId, item_index=residual_set$movieId, 
                              rating=residual_set$resid, index1=TRUE)
test_reco <- data_memory(user_index=test_set$userId, item_index=test_set$movieId, index1=TRUE)

# This step is lengthy and could take 15+ minutes
opts <- reco$tune(train_reco, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                              costp_l1=0, costq_l1=0,
                                              nthread = 1, niter = 10))

# Train the model of reco
reco$train(train_reco, opts = c(opts$min, nthread=1, niter=20))

# Predict results for the test set
reco_pred <- reco$predict(test_reco, out_memory())
preds <- cbind(test_set, reco_pred) %>% 
  left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(pred = mu + b_m + b_u + b_g + reco_pred) %>%
  pull(pred)
rmseMF <- RMSE(preds, test_set$rating)
RMSE_result4 <- bind_rows(RMSE_result3, data_frame(Model="Matrix Factorization", RMSE = rmseMF )) %>%
  arrange(desc(RMSE)) #arranging RMSE from largest to smallest to see best model
RMSE_result4

# Matrix Factorization was best model and will be applied to validation data
# Making sure userId and movieId in validation set are also in the train set
valid <- validation %>% 
  semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
valid_reco <- data_memory(user_index=valid$userId, item_index=valid$movieId, index1=TRUE)
r_pred <- reco$predict(valid_reco, out_memory())
val_preds <- cbind(valid, r_pred) %>% 
  left_join(movie_avg, by='movieId') %>% left_join(user_avg, by='userId') %>%
  left_join(genre_avg, by='genres') %>% mutate(pred = mu + b_m + b_u + b_g + r_pred) %>%
  pull(pred)
rmseFinal <- RMSE(val_preds, valid$rating)



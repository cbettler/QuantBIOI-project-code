
########################################################################################################
# Actual code implementation for group project

#Problem:
#Classification of cancer tumor type using different classification algorithms

# Step One
# read in the data and clean it

data <- read.csv(file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/cancer_data.csv")

#drop id and x column
data$id <- NULL
data <- data[,1:31]
data_feat <- colnames(data)

#check data structure
str(data)
summary(data$diagnosis)
#export cleaned cancer data as csv to python, so as to run the classification algorithm in python
#write.csv(data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/cancer_data.csv")

#Step 2, Reduced dimension of data using different dimensionality reduction technique

#Use PCA to reduce data dimension and retain over 95% variation
pca <- princomp(data[,2:31], cor = TRUE)
summary(pca, loadings=T)
#check the scree plot (cumulative variance)
plot(pca,type="l")

# We choose principal components 1 to 10 because they explain more than 95% of the variation in the cancer dataset.
pcs <- pca$scores[,1:10]

#combine diagnosis and pcs
pc_data <- cbind(as.character(data$diagnosis), pcs)
pc_data <- as.data.frame(pc_data)
names(pc_data)[names(pc_data) == "V1"] <- "diagnosis"
#export pca_data as csv to python, so as to run the classification algorithm in python
#write.csv(pc_data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/pc_data.csv")

library(ggbiplot)
g <- ggbiplot(pca, obs.scale = 1, var.scale = 1, groups = data[,1], ellipse = TRUE, circle = TRUE)
print(g)

# Determine most important features of the breast cancer dataset and use them to reduce the cancer data dimension


#Feature Importance Using Random Forest
library(caret)

# prepare training scheme
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

feature_imp <- function(model, title) {
  
  # estimate variable importance
  importance <- varImp(model, scale = TRUE)
  
  # prepare dataframes for plotting
  importance_df_1 <- importance$importance
  importance_df_1$group <- rownames(importance_df_1)
  
  importance_df_2 <- importance_df_1
  importance_df_2$Overall <- 0
  
  importance_df <- rbind(importance_df_1, importance_df_2)
  
  plot <- ggplot() +
    geom_point(data = importance_df_1, aes(x = Overall, y = group, color = group), size = 2) +
    geom_path(data = importance_df, aes(x = Overall, y = group, color = group, group = group), size = 1) +
    theme(legend.position = "none") +
    labs(
      x = "Importance",
      y = "",
      title = title,
      subtitle = "Scaled feature importance",
      caption = "\nDetermined with Random Forest and
      repeated cross validation (10 repeats, 10 times)"
    )
  
  return(plot)
  
}


imp_1 <- train(diagnosis ~ ., data = data, method = "rf", preProcess = c("scale", "center"), trControl = control)
p1 <- feature_imp(imp_1, title = "Breast cancer dataset")
print(p1)

#Select features with over 50 unit of importance. There are 5 features
imp_feat <- c("diagnosis", "area_worst", "concave.points_mean", "concave.points_worst", "perimeter_worst", "radius_worst")
rndf_filt_data <- subset(data, select = imp_feat)
#export rndf data as csv to python so as to run the classification algorithm in python
#write.csv(rndf_filt_data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/rndf_filt_data.csv")

###
#Use Recursive Feature Elimination method to select important features in the dataset
set.seed(1234)
# define the control using a random forest selection function with cross validation
rfcontrol <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# run the Recursive Feature Elimination algorithm
results_1 <- rfe(x = data[,2:31], y = as.factor(data$diagnosis), sizes = c(1:30), rfeControl = rfcontrol)

# chosen features
predictors(results_1)

#reduce the dimension of the cancer dataset by the features selected by RFE
rfe_filt_data <- subset(data, select = c("diagnosis", predictors(results_1)))
#export data as csv to python, so as to run the classification algorithm in python
#write.csv(rfe_filt_data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/rfe_filt_data.csv")

##
#Feature Importance Using Correlation matrix plot
#The idea is to remove features that are highly correlated, and thereby reduce data dimension and regard the remaining features
# as important features

#install.packages("corrplot")
library(corrplot)

# calculate correlation matrix
corMatMy <- cor(data[,2:31])
corrplot(corMatMy, order = "hclust")

#Apply correlation filter at 0.70,
highlyCor <- colnames(data[,2:31])[findCorrelation(corMatMy, cutoff = 0.7, verbose = TRUE)]

#view the features that are flagged for removal
highlyCor

#Then create a data filtered by corr
cor_filt_data <- subset(data, select = c("diagnosis", highlyCor))
#export data as csv to python, so as to run the classification algorithm in python
#write.csv(cor_filt_data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/cor_filt_data.csv")

####
#NEXT steps
#1 import data to python and split into train and test sets
#2 run classification algorithms on the train and test
#3 classification algorithms to use are: Logistic Regression, K nearest neighbour, support vector machines, Random Forest Classification
#4 use monte carlo (mc) simulation to generate data for the 5 most important feature from RF, and run Kruskal wallis on each mc feature 
# against original feature. Then select mc that have the largest p-value. Reason is beacuse we want mc and original features to 
#have no difference in mean.
#5 fit classification algorithm with mc data and test with mc data
#6 use mc data as train and original data as test. And Vice Versa
# calculate accuracy at each classification step


#### Monte Carlo was done in python
#create dataset for Monte Carlo using the random forest (rndf) filtered data
#Split the dataset into two. train and test. Use the train (mc_data) to generate Monte data. train with MC data and test on test
library(caret)

mt_data <- createDataPartition(rndf_filt_data$diagnosis, p=0.7, list = FALSE)
mc_data <- rndf_filt_data[mt_data,]
mc_test_data <- rndf_filt_data[-mt_data,]

#subset mc_data into M and B datasets. Then use this two dataframes to generate MC for all their features. This is to know the 
#diagnosis of each MC simulation value
mc_data_M <- subset(mc_data, mc_data$diagnosis == "M")
mc_data_B <- subset(mc_data, mc_data$diagnosis == "B")

#Take out the diagnosis column and store in Y, so that we can cbind it with the mc features generated from this mc_data
mc_data_M_Y <- mc_data_M$diagnosis
mc_data_B_Y <- mc_data_B$diagnosis

#drop the diagnosis columns of the mc data
mc_data_M$diagnosis <- NULL
mc_data_B$diagnosis <- NULL

#Export all the created MC data as csv to python, so as to run the monte carlo algorithm in python
#write.csv(mc_test_data, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/mc_test_data.csv")
#write.csv(mc_data_M, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/mc_data_M.csv")
#write.csv(mc_data_B, file = "C:/Users/okoro/OneDrive/Desktop/STAT 437/mc_data_B.csv")




#Bar Plotting for accuracy Performances of the classificaction algorithms on the original dataset and Dimensionally reduced dataset
#The accuracy values were generated in python and copied into R for ease

Al_names <- c("Logistic R", "SVM", "SVM Kernel", "Random F", "KNN")
colors = c("green","orange","brown", "red", "blue")

#Original Dataset with all features retained
accuracy <- c(0.956140350877193, 0.956140350877193, 0.956140350877193, 0.9122807017543859, 0.9298245614035088)
barplot(accuracy, names.arg = Al_names, col = colors, beside = TRUE, main = "Classification Algorithms Accuracy Scores")

#PCA filtered Data
accuracyPca <- c(0.956140350877193, 0.956140350877193, 0.9473684210526315, 0.8947368421052632, 0.9035087719298246)
barplot(accuracypca, names.arg = Al_names, col = colors, beside = TRUE, main = "Classification Algorithms Accuracy Scores on PCA filtered Data")

#RF filtered Data 
accuracyRF <- c(0.9473684210526315, 0.9385964912280702, 0.9298245614035088, 0.9122807017543859, 0.9210526315789473)
barplot(accuracyRF, names.arg = Al_names, col = colors, beside = TRUE, main = "Classification Algorithms Accuracy Scores on RF filtered Data")

#RFE filtered data
accuracyRFE <- c(0.956140350877193, 0.956140350877193, 0.9473684210526315, 0.8947368421052632, 0.9035087719298246)
barplot(accuracyRFE, names.arg = Al_names, col = colors, beside = TRUE, main = "Classification Algorithms Accuracy Scores on RFE filtered Data")

#cor filtered Data
accuracycor <- c(0.956140350877193, 0.956140350877193, 0.9298245614035088, 0.8947368421052632, 0.9385964912280702)
barplot(accuracycor, names.arg = Al_names, col = colors, beside = TRUE, main = "Classification Algorithms Accuracy Scores on COR filtered Data")

canData <- read.csv(file="/Users/Ian/Documents/GitHub/QuantBIOI-project-code/data.csv", header=TRUE, sep=",")

data = subset(data, select=-c("id"))


ncol(canData)

canData <- canData[2:ncol(canData)-1]


boxplot(canData$radius_mean)
        
boxplot(canData$area_mean)


canData$diagnosis <- ifelse(canData$diagnosis == "M",1, 0)

canData$diagnosis

is.na(canData)

modelCan <- glm(diagnosis ~ .,family = binomial(logit), data=canData)

summary(modelCan)

pr <- predict(modelCan, canData, type = "response")

canData$diagnosis

table(actual = canData$diagnosis, predicted = pr > .5 )

#----------------------------
library(MASS)

fit <- lda(diagnosis ~ ., data=canData, na.action="na.omit", CV=TRUE)

ct <- table(canData$diagnosis, fit$class)
diag(prop.table(ct, 1))

sum(diag(prop.table(ct)))

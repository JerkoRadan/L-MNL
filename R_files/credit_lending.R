###########################################################
#                                                         #
# Author: Jerko Radan Cruz                                #
# This is an extra file for the Master thesis project:    #
#'Interpretable neural networks for a multiclass          #
# assessment of credit risk'                              #
#                                                         #
# It run a Cumulative Logit model (polr).                 #
# Evaluates the Friedmantest and a pairwise comparison    #
# using Wilcoxon rank sum test                            #
#                                                         #
###########################################################


#install.packages("tram")
#install.packages("dplyr")
install.packages("caret")
install.packages("VGAM")
install.packages("brant")
install.packages("ggpubr")
install.packages("rstatix")

remotes::install_github("leifeld/texreg")

#library(ontram)
#library(tram)
library(MASS)
library(dplyr)
library(tidyverse)
library(caret)
library(texreg)
library(stats)
library(VGAM)
library(brant)

library(ggpubr)
library(rstatix)


# ------------------------ Cumulative logit model -----------------------------------------

train_df <- read.csv(file = 'Train.csv')
test_df <- read.csv(file = 'Test.csv')
head(train_df)

rm(f)

train_df$loan_status <- replace(train_df$loan_status, train_df$loan_status == "1", "Categ1")
train_df$loan_status <- replace(train_df$loan_status, train_df$loan_status == "2", "Categ2")
train_df$loan_status <- replace(train_df$loan_status, train_df$loan_status == "3", "Categ3")

test_df$loan_status <- replace(test_df$loan_status, test_df$loan_status == "1", "Categ1")
test_df$loan_status <- replace(test_df$loan_status, test_df$loan_status == "2", "Categ2")
test_df$loan_status <- replace(test_df$loan_status, test_df$loan_status == "3", "Categ3")

train_df$loan_status <- factor(train_df$loan_status, order = TRUE, levels = c("Categ1", "Categ2", "Categ3"))
test_df$loan_status <- factor(test_df$loan_status, order = TRUE, levels = c("Categ1", "Categ2", "Categ3"))
str(train_df)
str(test_df)

colnames(train_df)
colnames(test_df)

colnames(train_df)[15] <- "addr_state_Midwest_region"
colnames(train_df)[16] <- "addr_state_Northeast_region"
colnames(train_df)[17] <- "addr_state_Southern_region"
colnames(train_df)[18] <- "addr_state_Western_region"

colnames(test_df)[15] <- "addr_state_Midwest_region"
colnames(test_df)[16] <- "addr_state_Northeast_region"
colnames(test_df)[17] <- "addr_state_Southern_region"
colnames(test_df)[18] <- "addr_state_Western_region"

# Ordinal Logit Model (Proportional Odds)

m1 <- polr(loan_status ~ ., data = train_df[,-18], method = "logistic", Hess = TRUE)
#m2 <- Polr(loan_status ~ ., data = train_df, method = "logistic")
summary(m1)
coeff <- coef(summary(m1))
extract(m1)

brant(m1)

y_pred <- predict(m1, test_df[,-18], type = "p")

write.csv(y_pred, "C:\\Users\\jerko\\Documents2\\KU Leuven - otros cursos\\4. Thesis\\Experiment\\Lending Club\\credit_paper\\data\\Polr_y_pred.csv", row.names = FALSE)

folds <- createFolds(train_df$loan_status, 5)
str(folds)

i <- 1
for (f in folds) {
  m <- polr(loan_status ~ ., data = train_df[-f,-18], method = "logistic", Hess = TRUE)
  print(i)
  print(extract(m))
  pred <- predict(m, train_df[f,-18], type = "p")
  write.csv(train_df[f,37], paste(c("C:\\Users\\jerko\\Documents2\\KU Leuven - otros cursos\\4. Thesis\\Experiment\\Lending Club\\credit_paper\\data\\", i, "Fold_y_true.csv"), collapse = ""), row.names = FALSE)
  write.csv(pred, paste(c("C:\\Users\\jerko\\Documents2\\KU Leuven - otros cursos\\4. Thesis\\Experiment\\Lending Club\\credit_paper\\data\\", i, "Fold_y_pred.csv"), collapse = ""), row.names = FALSE)
  i <- i +1
}

# -------------------------------------------------------------------------------------

# Metrics results comparison ----------------------------------------------------------

results_met10 <- read.csv(file = 'Friedman_new_small_wBIC.csv')
results_met11 <- read.csv(file = 'Friedman_new_smallv3.csv')

levels(results_met11$Model)
levels(results_met11$Metric)

# Friedman test
sum_fried_test4 <-  friedman_test(Value ~ Model|Metric, data = results_met10)
sum_fried_test5 <-  friedman_test(Value ~ Model|Metric, data = results_met11)

# Wilcoxon rank sum test
res_wilcox_100_new_small_wBIC <- pairwise.wilcox.test(results_met10$Value, results_met10$Model, p.adjust.method = "BH")
res_wilcox_100_new_smallv3 <- pairwise.wilcox.test(results_met11$Value, results_met11$Model, p.adjust.method = "BH")


# ---------------------------- END -----------------------------------------------------





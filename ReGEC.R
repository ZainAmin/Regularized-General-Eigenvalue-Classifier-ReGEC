# Regularized General Eigenvalue Classifier (ReGEC) Classifier

# Load the required packages
# install.packages("e1071", dep = TRUE) 

library(MASS)
library(caret)
library(e1071)

# Function to handle data loading and pre-processing 
data_processing <- function(dataset_name) {
  switch(dataset_name, 
       cleveland={
         print("Load the Cleveland Heart Disease dataset")
         url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
         data <- read.csv(url, header = FALSE, na.strings = "?")
         
         # Checking how many rows is the dataset = 303
         print(paste("Number of Cleveland Heart Disease dataset rows is ", str(nrow(data))))
         
         # Assign column names to the dataset
         colnames(data) <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                             "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")
         
         # There are some rows of data that contain Na, based on the observation
         # we can drop them from the dataset
         data <- data[!(is.na(data$ca) | is.na(data$thal)),]
         print(paste("Number of rows remaining in ", dataset_name, " is ", nrow(data))) #303 - 6 = 297 rows remaining
         
         
         # Convert the target variable to a factor
         data$target <- factor(ifelse(data$target == 0, "negative", "positive"))
         
         return(data)
       },
       breast_cancer={
         print("Load Breast Cancer Dataset")
         url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
         data <- read.csv(url, header = FALSE)
         
         # Checking how many rows is the dataset
         print(paste("Number of Breast Cancer Dataset rows is ", nrow(data)))
         
         # Assign column names to the dataset
         colnames(data) <- c("class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", 
                             "breast", "breast-quad", "target")
         
         data$target <- factor(ifelse(data$target == "no", "negative", "positive"))
         
         return(data)
       },
       pima_indians={
         print("Load Pima Indians Dataset")
         url <- "https://query.data.world/s/2wlei2ucaxuvc54ib5yc5erd4acyiw?dws=00000"
         data <- read.csv(url, header = FALSE)
         
         # Checking how many rows is the dataset
         print(paste("Number of Pima Indians Dataset rows is ", nrow(data)))
         
         # Assign column names to the dataset
         colnames(data) <- c("pregnancy_time", "glucose_concentration", "blood_pressure", 
                             "skin_fold_thick", "serum_insulin", "body_mass", "diabetes pedigree", 
                             "age", "target")
         
         data$target <- factor(ifelse(data$target == 0, "negative", "positive"))
         
         return(data)
       },
       german={
         print("Load German Dataset")
         url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
         data <- read.csv(url, header = FALSE, sep = " ")
         
         # Checking how many rows is the dataset
         print(paste("Number of German Dataset rows is ", nrow(data)))
          
         # Assign column names to the dataset
         colnames(data) <- c("checking_account_status", "duration_months", "credit_history", "purpose", 
                                    "credit_amount", "savings_account_status", "employment_status", 
                                    "installment_rate", "personal_status_sex", "other_debtors", 
                                    "present_residence", "property", "age", "other_installment_plans", 
                                    "housing", "number_existing_credits", "job", "dependents", "phone", "foreign_worker", 
                                    "target")
         
         data$target <- factor(ifelse(data$target == 1, "positive", "negative"))
         
         return(data)
       }
  )
}

# Feature extraction using LDA approach - supervised learning approach
feature_extraction <- function(target, train_data, test_data) {
  # Fit an LDA model to the training data
  lda.fit <- lda(target ~ ., data= train_data)
  
  # Extract the LDA features from the training and testing data
  train_lda <- predict(lda.fit, train_data)
  test_lda <- predict(lda.fit, test_data)
  
  return(c(train_lda=train_lda, test_lda=test_lda))
}

# Predict function using RGEC classifier 
predict_RGEC_classifier <- function(model, projected_test_data) {
  # Make predictions on the projected data
  predictions <- predict(model, newdata = as.data.frame(projected_test_data))
  
  # Return the predicted labels
  return(predictions)
}

# Training RGEC Classifier function
train_RGEC_classifier <- function(data, kernel_type, dataset_name, lamda, gamma, train_size) {
  # Process the dataset
  data <- data_processing(dataset_name)
  
  # Train/test splitting the dataset based on the train_size
  set.seed(123)
  train_index <- sample(1:nrow(data), nrow(data) * train_size)
  train_data <- data[train_index,]
  test_data <- data[-train_index,]
  
  # LDA Feature Extraction
  # Best suited for class separation and uses within-class and between-class scatter matrices
  # Also used for dimensional reduction to help projecting large datasets onto a lower-dimensional space
  # with having good class-separability 
  # It also maximizes the component axes for class-separation
  lda <- feature_extraction(target=train_data$target, train_data, test_data)
  
  lda_train_features = lda$train_lda.x
  lda_test_features = lda$test_lda.x
  
  
  # Defining the regularization parameter to help prevent over-fitting.
  regularization_parameter <- lamda
  
  # Compute the overall mean and variance for train features
  train_mean <- colMeans(lda_train_features)
  train_conv <- cov(lda_train_features)
  
  # Compute the between-class and within-class scatter matrices
  train_within_class <- train_conv +
    diag(regularization_parameter, ncol(lda_train_features))
  train_between_class <- (t(train_mean - lda_train_features)
                          %*% (train_mean - lda_train_features)) / nrow(lda_train_features)
  
  # Compute the generalized eigenvectors and eigenvalues
  eigen <- eigen(solve(train_within_class) %*% train_between_class)
  
  # Sort the eigenvalues in descending order
  eigen_order <- order(eigen$values, decreasing = TRUE)
  eigen$values <- eigen$values[eigen_order]
  eigen$vectors <- eigen$vectors[, eigen_order]
  
  # Obtain the projection matrix - matrix d
  projection_matrix <- as.matrix(eigen$vectors)
  
  # Project the test data on the selected eigenvectors, to transform the sample 
  # onto the new subspace - d * k eigenvector matrix
  projected_train_data <- as.matrix(lda_train_features) %*% projection_matrix
  projected_test_data <- as.matrix(lda_test_features) %*% projection_matrix
  
  # Train SVM model on the projected training data using both kernels 
  # (linear and gaussian)
  if(kernel_type == 'linear'){
    print("Using SVM linear kernel type.")
    start_time <- Sys.time()
    
    svm_model <- svm(x = projected_train_data, y = train_data$target, 
                     kernel=kernel_type, cost=1)
    
    end_time <- Sys.time()
  }else{
    print("Using SVM gaussian kernel type.")
    start_time <- Sys.time()
    
    svm_model <- svm(x = projected_train_data, y = train_data$target, 
                     kernel=kernel_type, gamma=gamma)
    
    end_time <- Sys.time()
  }
  
  # Obtain the predictions 
  predictions <- predict_RGEC_classifier(svm_model, projected_test_data)
  
  # Obtain the confusion matrix to evaluate the model performance
  cm <- confusionMatrix(table(predictions, test_data$target))
  print(cm)
  
  # Obtain the elapsed time in seconds
  elapsed_time <- end_time - start_time
  print(elapsed_time)
}

# Running ReGEC classifier on Cleveland Heart Dataset using linear kernel
train_RGEC_classifier(data,
                      kernel_type = 'linear',
                      dataset_name= 'cleveland',
                      lamda=0.2,
                      train_size=0.9,
                      gamma=NA)


# Running ReGEC classifier on Pima Indians Dataset using linear kernel
# train_RGEC_classifier(data,
#                       kernel_type = 'linear',
#                       dataset_name= 'pima_indians',
#                       lamda=0.2,
#                       train_size=0.9,
#                       gamma=NA)
# 


# Running ReGEC classifier on Breast Cancer Dataset using radial gaussian kernel
# train_RGEC_classifier(data,
#                       kernel_type = 'radial',
#                       dataset_name= 'breast_cancer',
#                       lamda=0.001,
#                       train_size=0.7,
#                       gamma=50)

# Running ReGEC classifier on German Dataset using radial gaussian kernel
# train_RGEC_classifier(data,
#                       kernel_type = 'radial',
#                       dataset_name= 'german',
#                       lamda=0.001,
#                       train_size=0.7,
#                       gamma=500)




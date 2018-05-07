

#======================================
#     Big Project Part 2: hmeq.csv
#======================================

#--------------------------------------
#     IDA / Preprocessing Data 
#-------------------------------------

# Đọc dữ liệu: 
rm(list = ls())
library(tidyverse)
library(magrittr)

hmeq <- read.csv("D:/Teaching/data_science_banking/hmeq/hmeq.csv")

# Viết hàm điều tra tỉ lệ dữ liệu thiếu ở từng cột biến: 
ti_le_na <- function(x) {100*sum(is.na(x)) / length(x)}

# Kiểm tra NA: 
hmeq %>% summarise_all(funs(ti_le_na))

# Viết hàm thay thế  NA bằng mean cho biến liên tục: 

thay_na_mean <- function(x) {
  tb <- mean(x, na.rm = TRUE)
  x[is.na(x)] <- tb
  return(x)
}

# Viết hàm thay thế NA bằng lớp xuất hiện nhiều nhất
# cho biến định tính (đề phòng nếu cần dùng đến): 

thay_na_factor <- function(x) {
  u <- x %>% 
    table(useNA = "ifany") %>% 
    as.data.frame() %>% 
    arrange(-Freq)
  k <- as.character(u[, 1])
  x[is.na(x)] <- k[1]
  return(x)
}

# Test hàm: 

x <- c("A", "A", "B", "C", "A", NA)
thay_na_factor(x)

# Nghề nghiệp và lí do vay có một số không được đặt tên: 
table(hmeq$REASON)
table(hmeq$JOB)

# Nên ta viết hàm đặt tên lại cho lí do vay: 
name_job <- function(x) {
  x %<>% as.character()
  ELSE <- TRUE
  quan_tam <- c("Mgr", "Office", "Other", "ProfExe", "Sales", "Self")
  case_when(!x %in% quan_tam ~ "Other", 
            ELSE ~ x)
}

# Tương tự là cho lí do vay: 
name_reason <- function(x) {
  ELSE <- TRUE
  x %<>% as.character()
  case_when(!x %in% c("DebtCon", "HomeImp") ~ "Unknown", 
            ELSE ~ x)
}


# Xử lí số liệu thiếu và dán nhãn lại: 
hmeq_proce <- hmeq %>% 
  mutate_if(is.numeric, thay_na_mean) %>% 
  mutate_at("REASON", name_reason) %>% 
  mutate_at("JOB", name_job)

# Kiểm tra lại rằng không còn NA: 
hmeq_proce %>% summarise_all(funs(ti_le_na))

#Dán lại nhãn trong đó 1 = Bad, 0 = Good:
hmeq_proce %<>% mutate(BAD = case_when(BAD == 1 ~ "Bad",
                                       BAD == 0 ~ "Good"))

# Chuyển hóa bất kì cột biến nào ở character về factor: 
hmeq_proce %<>% mutate_if(is.character, as.factor)


# Tỉ lệ các loại hồ sơ: 
hmeq_proce %>% 
  group_by(BAD) %>% 
  count() %>% 
  mutate(Percent = 100*n / nrow(hmeq_proce)) %>% 
  knitr::kable()

# Thực  hiện  phân chia dữ liệu theo tỉ lệ 50 -  50: 
library(caret)
set.seed(29)
id <- createDataPartition(y = hmeq_proce$BAD, p = 0.5, list = FALSE)

train_data <- hmeq_proce[id, ]
test_data <- hmeq_proce[-id, ]


#--------------------------------------------------
#   Thực hiện và so sánh nhanh một số thuật toán
#--------------------------------------------------

# Thiết lập các chế độ kiểm tra cho mô hình bằng cross - validation: 

set.seed(1)
ctrl <- trainControl(method = "repeatedcv", 
                     # k = 5: 
                     number = 5, 
                     # Lặp lại 6 lần: 
                     repeats = 10,
                     # Lấy ra tất cả 12 tiêu chí đánh giá mô hình: 
                     summaryFunction = multiClassSummary, 
                     # Sử dụng tính toán song song: 
                     allowParallel = TRUE)


# Thiết lập các chế độ tính toán song song: 

library(doParallel)
n_cores <- detectCores()
registerDoParallel(cores = n_cores - 1)

# Logistic (chú ý rằng Sensitivity = BB / (BB + GB))

logistic <- train(BAD ~., 
                  data = train_data, 
                  method = "glm",
                  family = "binomial", 
                  trControl = ctrl, 
                  metric = "AUC", 
                  preProcess = NULL)

# Probit: 

probit <- train(BAD ~., 
                data = train_data, 
                method = "glm",
                family = "binomial"(link = "probit"), 
                trControl = ctrl, 
                metric = "AUC", 
                preProcess = NULL)


# Support Vector Machine (SVM): 

set.seed(1)
svm <- train(BAD ~., 
             data = train_data, 
             method = "svmRadial",
             trControl = ctrl, 
             metric = "AUC", 
             preProcess = c("scale"))



# Random Forest: 

set.seed(1)
rf <- train(BAD ~., 
            data = train_data, 
            method = "rf",
            trControl = ctrl, 
            metric = "AUC", 
            preProcess = c("scale"))


# KNN: 
set.seed(1)
knn <- train(BAD ~., 
            data = train_data, 
            method = "knn",
            trControl = ctrl, 
            metric = "AUC", 
            preProcess = c("scale"))


# Data Frame về kết quả của các mô hình này: 
result_df <- bind_rows(logistic$resample %>% 
                         select(-Resample) %>% 
                         mutate(Model = "Logistic"), 
                       probit$resample %>% 
                         select(-Resample) %>% 
                         mutate(Model = "Probit"), 
                       svm$resample %>% 
                         select(-Resample) %>% 
                         mutate(Model = "SVM"), 
                       rf$resample %>% 
                         select(-Resample) %>% 
                         mutate(Model = "RF"), 
                       knn$resample %>% 
                         select(-Resample) %>% 
                         mutate(Model = "knn"))


# RF là mô hình có ưu thế nhất: 
theme_set(theme_minimal())
result_df %>% 
  gather(metric, value, -Model) %>% 
  ggplot(aes(Model, value)) + 
  geom_boxplot(aes(fill = Model, color = Model), alpha = 0.3, show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")


result_df %>% 
  group_by(Model) %>% 
  summarise_each(funs(mean, median, sd, min, max, n()), Accuracy) %>% 
  arrange(-Accuracy_mean)


result_df %>% 
  group_by(Model) %>% 
  summarise_each(funs(mean, median, sd, min, max, n()), Sensitivity) %>% 
  arrange(-Sensitivity_mean)


result_df %>% 
  group_by(Model) %>% 
  summarise_each(funs(mean, median, sd, min, max, n()), Specificity) %>% 
  arrange(-Specificity_mean)

result_df %>% 
  group_by(Model) %>% 
  summarise_each(funs(mean, median, sd, min, max, n()), Kappa) %>% 
  arrange(-Kappa_mean)

result_df %>% 
  ggplot(aes(Model, Accuracy)) + 
  geom_boxplot()


#-----------------------------------
#  Giải thích về Confusion Matrix  
#-----------------------------------


# Nếu muốn lấy kết quả dự báo với nhãn Bad hoặc Good: 
pred_lab <- predict(rf, newdata = test_data %>% select(-BAD), type = "raw")

# Nếu muốn lấy kết quả dự báo ở mức xác suất: 
pred_prob <- predict(rf, newdata = test_data %>% select(-BAD), type = "prob")

# Lập một DF so sánh kết quả dự báo của mô hình với thực tế: 

df_comp <- data.frame(du_bao = pred_lab, thuc_te = test_data$BAD)

# Lọc ra những hồ sơ nào được dự báo với nhãn Good: 
df_comp_filted <- df_comp %>% filter(du_bao == "Good")

# và so sánh: 
df_comp_filted$thuc_te %>% table()

# So sánh với ma trận nhầm lẫn: 
confusionMatrix(test_data$BAD, pred_lab, positive = "Bad")



#-----------------------------
# Hàm tính kết quả phân loại
#-----------------------------

test_model <- function(model, so_lan_lay_mau, so_ho_so, du_lieu) {
  ket_qua <- data.frame()
  for (i in 1:so_lan_lay_mau) {
    set.seed(i)
    id <- createDataPartition(y = du_lieu$BAD, 
                              p = so_ho_so / nrow(du_lieu), 
                              list = FALSE)
    test <- du_lieu[id, ]
    
    predicted <- predict(model, test, type = "raw")
    u <- table(test$BAD, predicted) %>% as.vector()
    ket_qua <- rbind(ket_qua, u)
    names(ket_qua) <- c("BB", "GB", "BG", "GG")
  }
  return(ket_qua)
}

# Sử dụng hàm: 
df_rf <- test_model(rf, 100, 1000, test_data)
df_knn <- test_model(knn, 100, 1000, test_data)


n_vay_tot <- df_rf$GG %>% sum()
n_vay_xau <- df_rf$BG %>% sum()

khoan_vay <- hmeq_proce$LOAN


so_tien_cho_vay_tot <- sample(khoan_vay, n_vay_tot, replace = TRUE)
so_tien_cho_vay_xau <- sample(khoan_vay, n_vay_xau, replace = TRUE)

loi_nhuan <- sum(0.1*so_tien_cho_vay_tot) - sum(so_tien_cho_vay_xau)
loi_nhuan

# Viết hàm mô phỏng lợi nhuận: 
profit_simu <- function(data_from_model, rate, so_lan_mo_phong) {
  n_vay_tot <- data_from_model$GG %>% sum()
  n_vay_xau <- data_from_model$BG %>% sum()
  prof <- c()
  
  for (i in 1:so_lan_mo_phong) {
    so_tien_cho_vay_tot <- sample(khoan_vay, n_vay_tot, replace = TRUE)
    so_tien_cho_vay_xau <- sample(khoan_vay, n_vay_xau, replace = TRUE)
    
    loi_nhuan <- sum(rate*so_tien_cho_vay_tot) - sum(so_tien_cho_vay_xau)
    prof <- c(prof, loi_nhuan)
  }
  return(prof)
  
}

# Sử dụng hàm: 
profit_simu(data_from_model = df_rf, 
            rate = 0.1, 
            so_lan_mo_phong = 5000) -> p1


profit_simu(data_from_model = df_knn, 
            rate = 0.1, 
            so_lan_mo_phong = 5000) -> p2


# Tạo ra DF về lợi nhuận: 

df_for_comp <- bind_rows(data_frame(Profit = p1, Model = rep("RF", length(p1))), 
                         data_frame(Profit = p2, Model = rep("KNN", length(p2))))

df_for_comp %>% 
  group_by(Model) %>% 
  summarise_each(funs(mean, median, min, max, sd), Profit)

df_for_comp %>% 
  ggplot(aes(Profit / 1000000)) + 
  geom_density(fill = "red", color = "red", alpha = 0.3) + 
  geom_histogram(aes(y = ..density..), color = "blue", fill = "blue", alpha = 0.3) + 
  facet_wrap(~ Model, scales = "free")




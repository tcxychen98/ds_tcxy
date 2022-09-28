# import packages & data set
library(readxl)
library(ggplot2)
library(gridExtra)
library(psy)
library(MASS)
library(psych)

whdata <- read.csv("WHR2019.csv", header = TRUE)
whdata_raw <- read_xls("WHR2019 Raw Data.xls", sheet="Figure2.6")  # to add dystopia index and residual
continents <- read.csv("Continents.csv", header = TRUE, fileEncoding="UTF-8-BOM")  # to add continent to each country

wh_new <- merge(whdata,whdata_raw[, c("Country","Dystopia (1.88) + residual")], by="Country")
wh_new <- merge(whdata,continents[, c("Country","Continent")], by="Country")
View(wh_new)

# Structure of each data
wh_new$Continent <- as.factor(wh_new$Continent)
str(wh_new)

# Summary Statistics
summary(wh_new[3:10])

#correlation plots
pairs(wh_new[3:9], main="Correlation Plot of All Continuous Variables")

######### Box plots
bp1 <- ggplot(stack(wh_new[4:9]), aes(x=ind,y=values)) +
  geom_boxplot() +
  labs(y="Score", x="Variables", title="Boxplot of Overall Scores Across Variables")

bp1 # Box plot for each variable regardless of countries or continents

# Box plots for each variables based on continents
gdp <- ggplot(data = wh_new, aes(x = Continent, y = GDP_per_capita, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs GDP Per Capita")

soc_supp <- ggplot(data = wh_new, aes(x = Continent, y = Social_support, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs Social Support")

life_exp <- ggplot(data = wh_new, aes(x = Continent, y = Life_expectancy, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs Life Expectancy")

free <- ggplot(data = wh_new, aes(x = Continent, y = Freedom, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs Freedom")

genero <- ggplot(data = wh_new, aes(x = Continent, y = Generosity, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs Generosity")

corrupt <- ggplot(data = wh_new, aes(x = Continent, y = Corruption, fill=Continent)) +
  geom_boxplot() +
  ggtitle("Boxplot of Continents vs Corruption")

grid.arrange(gdp, soc_supp, life_exp, free, genero, corrupt, ncol=2, nrow=3)

# Violin Plot of Score based on continent
ggplot(wh_new, aes(Continent, Score, fill=Continent)) + 
  geom_violin(aes(color = Continent), trim = T)+
  scale_y_continuous("Score", breaks= seq(0,30, by=.5))+
  geom_boxplot(width=0.4)+
  theme(legend.position="right") +
  ggtitle("Total Happiness Score Based On Continents")

####### PCA Analysis
wh_new2 <- wh_new[,-1]
rownames(wh_new2) <- wh_new[,1] #change row names

whdat <- wh_new2[2:8]
pr.out1 = prcomp(whdat, scale = TRUE)
pr.out1$rotation   #loadings

#biplot 
biplot(pr.out1 , scale =0, col=c(9,4), cex = 0.6,xlim=c(-4.5,4.5), main="Biplot of World Happiness")

#proportion of variance
summary(pr.out1)
pr.var1 =pr.out1$sdev^2 
pr.var1

pve1=pr.var1/sum(pr.var1)
pve1

#scree plot
plot(pve1 , main="Scree Plot of WHR2019",xlab=" Principal Component ", ylab=" Proportion of Variance Explained ", ylim=c(0,1) ,type="b")


####### Clustering
set.seed(18098392)
km1 = kmeans(pr.out1$x[,c(1:2)],centers=4,nstart=20)
km1

plot(pr.out1$x[, 1:2], type="n", main = "Scatterplot of Countries with k Clusters") +
text(pr.out1$x[, 1:2], rownames(whdat), col=(km1$cluster+1), cex = 0.6)



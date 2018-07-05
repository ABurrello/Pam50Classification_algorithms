clear all
close all
working_directory = sprintf ('C:/Users/ale_m/OneDrive/Desktop/Bioinfo/gcdDataset/');
[~,~,Dataset] = xlsread('C:/Users/ale_m/OneDrive/Desktop/Bioinfo/dataset1.xls')

for i = 2:1223
    A=sprintf(char(Dataset(i,1)));
    B=strcat(working_directory,A);
    C=readtable(B);
    D(:,i)=table2array(C(:,2));
    fprintf('%d\n',i);
end
Transcriptome = D';
C=readtable(B);
Transcriptome_labels=table2array(C(:,1));

%% 
[~,~,Dataset2] = xlsread('C:/Users/ale_m/OneDrive/Desktop/Bioinfo/Dataset_Labels.xls')

fatto = 0
clear Y
for i = 2:1223
    for j = 2:826
        if strcmp(char(Dataset(i,2)), char(Dataset2(j,1)))
            switch char(Dataset2(j,2))
                case 'HER2-enriched'
                    Y(i,:) = sprintf ('HER2-enriched');
                case 'Luminal A'
                    Y(i,:) = sprintf ('Luminal A    ');
                case 'Basal-like'
                    Y(i,:) = sprintf ('Basal-like   ');
                case 'Luminal B'
                    Y(i,:) = sprintf ('Luminal B    ');
                case 'Normal-like'
                    Y(i,:) = sprintf ('Normal-like  ');  
                case 'NA'
                    Y(i,:) = sprintf ('NA           ');                      
            end
            fatto = 1
        end
    end
    if fatto == 0
        if contains(char(Dataset(i,2)),'-11')
            Y(i,:) = sprintf('Healty       ');
        else
            Y(i,:) = sprintf('Not present  ');
        end
    end
    fatto = 0;
end
%% 
[~,~,Dataset2] = xlsread('C:/Users/ale_m/OneDrive/Desktop/Bioinfo/Dataset_Labels2.xls')
for i = 2:1223
    for j = 2:1149
        if strcmp(char(Dataset(i,2)), char(Dataset2(j,1))) && (strcmp(Y(i,:),'Not present  ') || strcmp(Y(i,:),'NA           '))
            switch char(Dataset2(j,2))
                case 'HER2-enriched'
                    Y(i,:) = sprintf ('HER2-enriched');
                case 'Luminal A'
                    Y(i,:) = sprintf ('Luminal A    ');
                case 'Basal-like'
                    Y(i,:) = sprintf ('Basal-like   ');
                case 'Luminal B'
                    Y(i,:) = sprintf ('Luminal B    ');
                case 'Normal-like'
                    Y(i,:) = sprintf ('Normal-like  ');  
                case 'NA'
                    Y(i,:) = sprintf ('NA           ');                      
            end
        end
    end

end

%% 

[~,~,Dataset2] = xlsread('C:/Users/ale_m/OneDrive/Desktop/Bioinfo/Dataset_Labels3.xls')
for i = 2:1223
    for j = 2:467
        if strcmp(char(Dataset(i,2)), char(Dataset2(j,1))) && (strcmp(Y(i,:),'Not present  ') || strcmp(Y(i,:),'NA           '))
            switch char(Dataset2(j,2))
                case 'HER2-enriched'
                    Y(i,:) = sprintf ('HER2-enriched');
                case 'Luminal A'
                    Y(i,:) = sprintf ('Luminal A    ');
                case 'Basal-like'
                    Y(i,:) = sprintf ('Basal-like   ');
                case 'Luminal B'
                    Y(i,:) = sprintf ('Luminal B    ');
                case 'Normal-like'
                    Y(i,:) = sprintf ('Normal-like  ');  
                case 'NA'
                    Y(i,:) = sprintf ('NA           ');                      
            end
        end
    end

end

filename = 'DatabaseFinal.xlsx';
sheet = 1;
xlRange = 'C3';
xlswrite(filename,D,sheet,xlRange)

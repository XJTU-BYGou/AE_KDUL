function [ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] = importSSCData(filename, dataLines)
%IMPORTFILE Import data from a file
%
%  Example:
%  [ssc_Time, ssc_Displacement, ssc_Force, ssc_Strain, ssc_Stress] = importfile(".\PSS-1-tension test-3-z1-AE-20190217.is_tens_RawData\Specimen_RawData_2.csv", [3, Inf]);


%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [3, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 5);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["ssc_Time", "ssc_Displacement", "ssc_Force", "ssc_Strain", "ssc_Stress"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable(filename, opts);

%% Convert to output type
ssc_Time = tbl.ssc_Time;
ssc_Displacement = tbl.ssc_Displacement;
ssc_Force = tbl.ssc_Force;
ssc_Strain = tbl.ssc_Strain;
ssc_Stress = tbl.ssc_Stress;
end
# %%
import numpy as np
import os
import pandas as pd
import sys
# %%
input_dir = sys.argv[1]
scenario_year = sys.argv[2]

# %%
households_file = os.path.join(input_dir, f'synthetic_households_{scenario_year}_base.csv')
persons_file = os.path.join(input_dir, f'synthetic_persons_{scenario_year}_base.csv')
landuse_file = os.path.join(input_dir, f'mgra15_based_input_{scenario_year}_base.csv')
#some columns are moved over from 2019 land use file
landuse2019_file = os.path.join(input_dir, 'mgra15_based_input2019.csv')


# %%
def process_household()-> pd.DataFrame:
    '''
        Series15 synthetic population (ABM3) does not contain all of the same columns as ABM2+ syn pop.
        This function will tranform the Series 15 to match what is expected.
        
        If these column names are changed, then the corresponding names need to be changed in the configs.
    '''
    households = pd.read_csv(households_file)
    xref_taz_mgra = pd.read_csv(landuse_file)[['mgra','taz']]
    
    # series 15 names to previous ABM2+ column names
    households_rename_dict = {
        'household_id': 'hhid',
        'HHADJINC': 'hinc',
        'VEH': 'veh',
        'NP': 'persons',
        'HHT': 'hht',
        'BLD': 'bldgsz'}
    households = households.rename(columns=households_rename_dict)
    
    #get taz of household location
    households = households.merge(xref_taz_mgra, on='mgra')
    
    #create household serial number; version
    households['household_serial_no'] = 0
    households['version'] = 0
    
    #create poverty (ref: ASPE)
    households['poverty_guideline'] = 12760
    households.loc[(households['persons'] == 2), 'poverty_guideline'] = 17240
    households.loc[(households['persons'] == 3), 'poverty_guideline'] = 21720
    households.loc[(households['persons'] == 4), 'poverty_guideline'] = 26200
    households.loc[(households['persons'] == 5), 'poverty_guideline'] = 30680
    households.loc[(households['persons'] == 6), 'poverty_guideline'] = 35160
    households.loc[(households['persons'] == 7), 'poverty_guideline'] = 39640
    households.loc[(households['persons'] == 8), 'poverty_guideline'] = 44120
    households.loc[(households['persons'] >= 9), 'poverty_guideline'] = 44120 + 4480* (households['persons']-8)
    
    households['poverty'] = households['hinc']/households['poverty_guideline']
    
    #create household income category
    conditions = [(households["hinc"] < 30000),
                  ((households["hinc"] >= 30000) & (households["hinc"] < 60000)),
                  ((households["hinc"] >= 60000) & (households["hinc"] < 100000)),
                  ((households["hinc"] >= 100000) & (households["hinc"] < 150000)),
                  (households["hinc"] >= 150000)]
    choices = [1,2,3,4,5]
    households["hinccat1"] = pd.Series(np.select(conditions, choices, default=1), dtype="int")
    
    #create number of workers in household
    households['hworkers'] = households['WIF']
    households.loc[(households['WIF']==3) & (households['persons']>=3) & (households['HUPAC']>=4), 'hworkers'] = households['persons']
    
    #fill NaN with o
    households['hworkers'].fillna(0, inplace=True)
    households['veh'].fillna(0, inplace=True)
    
    #create household unit type
    households['unittype'] = households['GQ_type']
    households.loc[(households['GQ_type'].isin([1,2,3])), 'unittype'] = 1
    
    #integer type of fields
    households['unittype']= households['unittype'].astype(int)
    households['hht']= households['hht'].astype(int)
    households['bldgsz']= households['bldgsz'].astype(int)
    households['hworkers']= households['hworkers'].astype(int)
    households['veh']= households['veh'].astype(int)
    households['hinc']= households['hinc'].astype(int)
    
    return households[["hhid",
                       "household_serial_no",
                       "taz",
                       "mgra",
                       "hinccat1",
                       "hinc",
                       "hworkers",
                       "veh",
                       "persons",
                       "hht",
                       "bldgsz",
                       "unittype",
                       "version",
                       "poverty"]].sort_values(by=['hhid'])

# %%
def process_persons()-> pd.DataFrame:
    '''
        Series15 synthetic population (ABM3) does not contain all of the same columns as ABM2+ syn pop.
        This function will tranform the Series 15 to match what is expected.
        
        If these column names are changed, then the corresponding names need to be changed in the configs.
    '''
    persons = pd.read_csv(persons_file, engine='pyarrow')
    
    # series 15 names to previous ABM2+ column names
    persons_rename_dict = {
        'household_id': 'hhid',
        'SPORDER': 'pnum',
        'AGEP': 'age',
        'SEX': 'sex',
        'RAC1P': 'rac1p',
        'WKW': 'weeks',
        'WKHP': 'hours',
        'SOC2': 'soc2'}
    
    persons.loc[:,'naics2_original_code'] = persons['NAICS2']
    persons = persons.rename(columns=persons_rename_dict).sort_values(by=['hhid','pnum'])
    
    #create household serial number; perid
    persons['household_serial_no'] = 0
    persons['version'] = 0
    persons['perid'] = range(1, 1+len(persons))
    
    PEMPLOY_FULL, PEMPLOY_PART, PEMPLOY_NOT, PEMPLOY_CHILD = 1, 2, 3, 4
    persons['pemploy'] = np.zeros(len(persons))
    persons['pemploy'] = np.where(persons.age < 16, PEMPLOY_CHILD, PEMPLOY_PART)
    persons['pemploy'] = np.where((persons.age >= 16) & ((persons.ESR == 3) | (persons.ESR == 6)), PEMPLOY_NOT, persons['pemploy'])
    persons['pemploy'] = np.where((persons.age>=16) & ((persons.ESR != 3) & (persons.ESR != 6)) & (persons.hours >= 35), PEMPLOY_FULL, persons['pemploy'])
    persons['pemploy'] = persons['pemploy'].astype(int)

    PSTUDENT_GRADE_OR_HIGH, PSTUDENT_UNIVERSITY, PSTUDENT_NOT = 1, 2, 3
    persons['pstudent'] = np.zeros(len(persons))
    persons['pstudent'] = np.where((persons.pemploy == 1) & (persons.age >= 16), PSTUDENT_NOT, persons.pstudent)
    persons['pstudent'] = np.where((persons.pemploy == 1) & (persons.age < 16), PSTUDENT_GRADE_OR_HIGH, persons.pstudent)
    persons['pstudent'] = np.where((persons.SCHG < 1) & (persons.age >= 16), PSTUDENT_NOT, persons.pstudent)
    persons['pstudent'] = np.where((persons.SCHG < 1) & (persons.age < 16), PSTUDENT_GRADE_OR_HIGH, persons.pstudent)
    persons['pstudent'] = np.where((persons.SCHG >= 15) & (persons.age >= 16) & (persons.pemploy != 1), PSTUDENT_UNIVERSITY, persons.pstudent)
    persons['pstudent'] = np.where((persons.SCHG >= 15) & (persons.age < 16) & (persons.pemploy != 1), PSTUDENT_GRADE_OR_HIGH, persons.pstudent)
    persons['pstudent'] = np.where((persons.age <= 19) & (persons.pemploy != 1) & (persons.SCHG >=1) & (persons.SCHG<=14), PSTUDENT_GRADE_OR_HIGH, persons.pstudent)
    persons['pstudent'] = np.where((persons.age > 19) & (persons.pemploy != 1) & (persons.SCHG >=1) & (persons.SCHG<=14),  PSTUDENT_UNIVERSITY, persons.pstudent)
    persons['pstudent'] = np.where(persons.pstudent == 0, 3, persons.pstudent)
    persons['pstudent'] = persons['pstudent'].astype(int)

    PTYPE_FULL, PTYPE_PART, PTYPE_UNIVERSITY, PTYPE_NONWORK, PTYPE_RETIRED, PTYPE_DRIVING, PTYPE_SCHOOL, PTYPE_PRESCHOOL = 1, 2, 3, 4, 5, 6, 7, 8
    persons['ptype'] = np.zeros(len(persons))
    persons['ptype'] = np.where((persons.pemploy == 1),  PTYPE_FULL, PTYPE_NONWORK)
    persons['ptype'] = np.where((persons.pstudent == 3) & (persons.pemploy == 2), PTYPE_PART, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 3) & (persons.age >= 65) & ((persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_RETIRED, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 3) & (persons.age < 6) & ((persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_PRESCHOOL, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 3) & (persons.age >= 6) & (persons.age <= 64) & ((persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_NONWORK, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 2)  & ((persons.pemploy == 2)  | (persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_UNIVERSITY, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 1) & (persons.age < 6)  & ((persons.pemploy == 2)  | (persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_PRESCHOOL, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 1) & (persons.age >= 16)  & ((persons.pemploy == 2)  | (persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_DRIVING, persons.ptype)
    persons['ptype'] = np.where((persons.pstudent == 1) & (persons.age >= 6) & (persons.age < 16)  & ((persons.pemploy == 2)  | (persons.pemploy == 3) | (persons.pemploy == 4)), PTYPE_SCHOOL, persons.ptype)
    persons['ptype'] = persons['ptype'].astype(int)
    
    #revise field of school grade of person: grade
    persons['grade'] = 0
    persons.loc[((persons['SCHG']>=2) & (persons['SCHG']<=10)), 'grade'] = 2
    persons.loc[((persons['SCHG']>=11) & (persons['SCHG']<=14)), 'grade'] = 5
    persons.loc[(persons['SCHG'].isin([15,16])), 'grade'] = 6
    
    #revise hispanic flag field: hisp
    persons['hisp'] = persons['HISP']
    persons.loc[(persons['HISP']!=1), 'hisp'] = 2
    
    #revise miltary field: hisp
    persons['miltary'] = 0
    persons.loc[(persons['MIL']==1), 'miltary'] = 1
    
    #revise educational attainment: educ
    persons['educ'] = np.where(persons.age >= 18, 9, 0)
    persons['educ'] = np.where(persons.age >= 22, 13, persons.educ)
    
    #employment related fields
    persons['occen5'] = 0
    persons['indcen'] = 0
    persons.loc[(persons['NAICS2'] == 'MIL'), 'indcen'] = 9770
    
    conditions3 = [((persons["NAICS2"] == '0') | (persons["NAICS2"] == '99')),
                   (persons["NAICS2"].isin(['21','23','48','49','4M','52','53','54','55','56'])),
                   (persons["NAICS2"].isin(['51','61','62','92'])),
                   (persons["NAICS2"].isin(['42','44','45','721','722'])),
                   (persons["NAICS2"].isin(['11','81'])),
                   (persons["NAICS2"].isin(['22','31','32','33','3M','71'])),
                   (persons["NAICS2"] == 'MIL')
                   ]
    choices3 = ['00-0000', '11-1021', '31-1010', '41-1011', '45-1010','51-1011', '55-1010']
    persons["occsoc5"] = pd.Series(np.select(conditions3, choices3, default='00-0000'), dtype="str")
    
    #integer type of fields
    persons['weeks']= persons['weeks'].astype(int)
    persons['hours']= persons['hours'].astype(int)
    
    return persons[["hhid",
                    "perid",
                    "household_serial_no",
                    "pnum",
                    "age",
                    "sex",
                    "miltary",
                    "pemploy",
                    "pstudent",
                    "ptype",
                    "educ",
                    "grade",
                    "occen5",
                    "occsoc5",
                    "indcen",
                    "weeks",
                    "hours",
                    "rac1p",
                    "hisp",
                    "version",
                    'naics2_original_code',
                    "soc2"]].sort_values(by=['hhid','pnum','perid'])

def process_landuse()-> pd.DataFrame:
    '''
        Series15 land use (ABM3) does not contain all of the same columns as ABM2+ land use.
        This function will tranform the Series 15 to match what is expected.
        
        If these column names are changed, then the corresponding names need to be changed in the configs.
    '''
    df_mgra = pd.read_csv(landuse_file)
    df2019_mgra = pd.read_csv(landuse2019_file)
    
    # series 15 names to previous ABM2+ column names
    landuse_rename_dict = {
        'zip': 'zip09',
        #'emp_tot':'emp_total',
        'majorcollegeenroll_total': 'collegeenroll',
        'othercollegeenroll_total': 'othercollegeenroll',
        'acre': 'acres',
        'landacre':'land_acres',
        'LUZ':'luz_id',
        'emp_tot': 'emp_total'}
    df_mgra = df_mgra.rename(columns=landuse_rename_dict)
      
    #attributes copy from 2019 land use file
    att_2019 = ['ech_dist', 'hch_dist', 'adultschenrl', 'hstallsoth', 'hstallssam', 'hparkcost', 'numfreehrs', 'dstallsoth',
                'dstallssam', 'dparkcost', 'mstallsoth', 'mstallssam', 'mparkcost', 'parkarea', 'budgetroom', 'economyroom',
                'luxuryroom', 'midpriceroom', 'upscaleroom', 'truckregiontype', 'MicroAccessTime', 'remoteAVParking',
                'refueling_stations', 'totint', 'duden', 'empden', 'popden', 'retempden', 'totintbin', 'empdenbin',
                'dudenbin', 'PopEmpDenPerMi']
    df_mgra[att_2019] = df2019_mgra[att_2019]
    
    return df_mgra

# %%
process_household().to_csv(os.path.join(input_dir, 'households.csv'), index=False)
process_persons().to_csv(os.path.join(input_dir, 'persons.csv'), index=False)
process_landuse().to_csv(os.path.join(input_dir, f'mgra15_based_input{scenario_year}.csv'), index=False)
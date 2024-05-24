# Import libraries

from scipy.stats import spearmanr
from scipy.stats import pearsonr

import pandas as pd

# Constants

region_dict = {
              'Dentate_nucleus': 'Dentate nucleus',
              'GP': 'Globus pallidus',
              'STN': 'Subthalamic nucleus',
              'STR': 'Striatum',
              'cingulate': 'Cingulate',
              'frontal': 'Pre-frontal',
              'occipital': 'Occipital',
              'parietal': 'Parietal',
              'premo motor': 'Pre-motor',
              'primary motor': 'Primary motor',
              'primary somatosensory': 'Primary somatosensory',
              'temporal': 'Temporal'
              }


# Helper functions


def extract_files(prediction_path, prediction_txt_file):
    """
    To extract prediction files & put in a list
    from the given
    'prediction_path/prediction_txt_file'
    """
    f = pd.read_csv(prediction_path + prediction_txt_file,
                    sep="\t",
                    header=None)
    f = f.rename(columns={0: 'file_name'})
    mylist = f['file_name']
    print("Read in: ", len(mylist), "files")

    files = []
    for i in mylist:
        dat = pd.read_csv(prediction_path + i, sep="\t")
        files.append(dat)
    print("Extracted:", len(files), "files")
    return files


def extract_area(csv_file_path, list_area_name):
    """
    To extract area of interest from areas of multiple objects
    """
    area = pd.read_csv(csv_file_path, sep=",")
    area.loc[:, 'Slice_ID'] = ['S'+i[0:6] for i in area['Image']]
    area_roi = area[(area['Name'].isin(list_area_name))]
    area_output = area_roi[['Slice_ID',
                            'Name',
                            'Area µm^2']].drop_duplicates(subset=['Slice_ID','Name'])
    area_output = area_output.reset_index(drop=True)
    return area_output


def calculate_tau_metrics(list_of_df, name):
    """
    Using a list of prediction dataframes,
    this function calculate basic metrics:
    counts and area for each tau burden type
    """
    summary = []
    for df in list_of_df:
        slide_name = 'S' + df['Image'][0][:-4]
        summary.append([slide_name,
                        name,
                        # tau counts
                        df[df['Class'] == 'Non_tau'].shape[0],
                        df[df['Class'] == 'Others'].shape[0],
                        df[df['Class'] == 'CB'].shape[0],
                        df[df['Class'] == 'NFT'].shape[0],
                        df[df['Class'] == 'Ambiguous'].shape[0],
                        df[df['Class'] == 'TA'].shape[0],
                        # area summed
                        df[df['Class'] == 'Non_tau']['Area µm^2'].sum(),
                        df[df['Class'] == 'Others']['Area µm^2'].sum(),
                        df[df['Class'] == 'CB']['Area µm^2'].sum(),
                        df[df['Class'] == 'NFT']['Area µm^2'].sum(),
                        df[df['Class'] == 'Ambiguous']['Area µm^2'].sum(),
                        df[df['Class'] == 'TA']['Area µm^2'].sum()
                        ])
    output = pd.DataFrame(data=summary,
                          columns=[
                            'Slice_ID',
                            'Name',
                            'Non_tau',
                            'Others',
                            'CB',
                            'NFT',
                            'Ambiguous',
                            'TA',
                            'Non_tau_area',
                            'Others_area',
                            'CB_area',
                            'NFT_area',
                            'Ambiguous_area',
                            'TA_area'
                                  ])
    return output


def calculate_tau_metrics_BG(list_of_df):
    """
    Using a list of prediction dataframes,
    this function calculate basic metrics:
    counts and area for each tau burden type
    for BG side which has 3 nuclei (Parent)
    """
    summary = []
    for df in list_of_df:
        slide_name = 'S' + df['Image'][0][:-4]
        regions = list(set(df['Parent']))
        for r in regions:
            roi = df[df['Parent'] == r]

            summary.append([slide_name,
                            r,
                            # tau counts
                            roi[roi['Class'] == 'Non_tau'].shape[0],
                            roi[roi['Class'] == 'Others'].shape[0],
                            roi[roi['Class'] == 'CB'].shape[0],
                            roi[roi['Class'] == 'NFT'].shape[0],
                            roi[roi['Class'] == 'Ambiguous'].shape[0],
                            roi[roi['Class'] == 'TA'].shape[0],
                            # area summed
                            df[df['Class'] == 'Non_tau']['Area µm^2'].sum(),
                            df[df['Class'] == 'Others']['Area µm^2'].sum(),
                            df[df['Class'] == 'CB']['Area µm^2'].sum(),
                            df[df['Class'] == 'NFT']['Area µm^2'].sum(),
                            df[df['Class'] == 'Ambiguous']['Area µm^2'].sum(),
                            df[df['Class'] == 'TA']['Area µm^2'].sum()
                            ])
    output = pd.DataFrame(data=summary,
                          columns=[
                            'Slice_ID',
                            'Name',
                            'Non_tau',
                            'Others',
                            'CB',
                            'NFT',
                            'Ambiguous',
                            'TA',
                            'Non_tau_area',
                            'Others_area',
                            'CB_area',
                            'NFT_area',
                            'Ambiguous_area',
                            'TA_area'
                                  ])
    return output


def calculate_density_areafrac(df_tau, df_area):
    """"
    Calculates tau density and area fraction
    """
    comb_ = df_tau.merge(df_area, on=['Slice_ID','Name'], how='left')

    comb = comb_.copy()
    comb.loc[:, 'Total_tau'] = comb['Others']+comb['CB']+comb['NFT']+comb['TA']
    comb.loc[:, 'Total_tau_hallmarks'] = comb['CB']+comb['NFT']+comb['TA']

    # density
    comb.loc[:, 'Total_tau_density'] = (comb['Total_tau'])/comb['Area µm^2']
    comb.loc[:, 'Tau_hallmark_density'] = (comb['Total_tau_hallmarks'])/comb['Area µm^2']
    comb.loc[:, 'NFT_density'] = comb['NFT']/comb['Area µm^2']
    comb.loc[:, 'CB_density'] = comb['CB']/comb['Area µm^2']
    comb.loc[:, 'TA_density'] = comb['TA']/comb['Area µm^2']
    comb.loc[:, 'Others_density'] = comb['Others']/comb['Area µm^2']
    comb.loc[:, 'Ambiguous_density'] = comb['Ambiguous']/comb['Area µm^2']

    # area fraction
    comb.loc[:, 'Total_tau_AF'] = (comb['NFT_area']+comb['CB_area']+comb['TA_area']+comb['Others_area'])/comb['Area µm^2']
    comb.loc[:, 'Tau_hallmarks_AF'] = (comb['NFT_area']+comb['CB_area']+comb['TA_area'])/comb['Area µm^2']
    comb.loc[:, 'NFT_AF'] = comb['NFT_area']/comb['Area µm^2']
    comb.loc[:, 'CB_AF'] = comb['CB_area']/comb['Area µm^2']
    comb.loc[:, 'TA_AF'] = comb['TA_area']/comb['Area µm^2']
    comb.loc[:, 'Others_AF'] = comb['Others_area']/comb['Area µm^2']
    comb.loc[:, 'Ambiguous_AF'] = comb['Ambiguous_area']/comb['Area µm^2']
    return comb

# helper functions for obj-semi


def data_inspection(dat):
    """
    Creates 3 tables summary of:
    1. Slides / region / stage
    2. No. of subjects / region
    3. No. of subjects / PSP subtype
    Prints No. of unique subjects
    """

    stages = list(set(dat['Stage_SK']))
    regions = list(set(dat['region_name']))

    s_r = []
    # Create table 1)
    for s in stages:  # for each PSP stage
        s_dat = dat[dat['Stage_SK'] == s]
        for_all_regions = []

        for r in regions:  # for each region
            # get number of slides
            r_ = s_dat[s_dat['region_name'] == r].shape[0]
            for_all_regions.append(r_)

        s_r.append(for_all_regions)

    output1 = pd.DataFrame(data={
                                'Stage 2': s_r[0],
                                'Stage 3': s_r[1],
                                'Stage 4': s_r[2],
                                'Stage 5': s_r[3],
                                'Stage 6': s_r[4]
                                })
    output1.insert(0, 'Regions', regions)

    # Creates table 2)
    PSP_subs = dat[['Patient_ID', 'Stage_SK']]
    PSP_subs = PSP_subs.drop_duplicates(subset=['Patient_ID'])
    PSP_sk = PSP_subs['Stage_SK'].value_counts()
    output2_ = PSP_sk.rename_axis('Stage_SK').reset_index(name='Counts')
    output2 = output2_.sort_values(by='Stage_SK')

    # Creates table 3)
    PSP_subs = dat[['Patient_ID', 'MDS-PSP last visit']]
    PSP_subs = PSP_subs.drop_duplicates(subset=['Patient_ID'])
    PSP_subtypes = PSP_subs['MDS-PSP last visit'].value_counts()
    output3 = PSP_subtypes.rename_axis('PSP subtype').reset_index(name='Counts')

    print('No. of unique patients: ', len(list(set(dat['Patient_ID']))))
    return output1, output2, output3


def correlation_tau(dat, tau_type):
    """
    Calculates correlation table
    between tau burden types.
    """
    tau_types = ['Total tau',
                 'Tau hallmark',
                 'CB',
                 'NFT',
                 'TA',
                 'TF'
                 ]

    corr_p = [pearsonr(dat[i], dat[tau_type]) for i in tau_types]   # get both r, p
    r_val = [round(i[0], 3) for i in corr_p]  # separates r out
    p_val = [i[1] for i in corr_p]  # separates p out
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                tau_type: r_val,
                                'p_val': p_val
                                })
    return output


def correlation_tau_noTA(dat, tau_type):
    """
    Calculates correlation table
    between tau burden types.
    """
    tau_types = ['Total tau',
                 'Tau hallmark',
                 'CB',
                 'NFT',
                 'TF'
                 ]

    corr_p = [pearsonr(dat[i], dat[tau_type]) for i in tau_types]   # get both r, p
    r_val = [round(i[0], 3) for i in corr_p]  # separates r out
    p_val = [i[1] for i in corr_p]  # separates p out
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                tau_type: r_val,
                                'p_val': p_val
                                })
    return output


def correlation_table(dat, stage):
    """
    Calculates correlation table
    between tau burden & stage rating e.g. Stage_SK.
    """
    tau_types = ['Total_tau_density',
                 'Tau_hallmark_density',
                 'CB_density',
                 'NFT_density',
                 'TA_density',
                 'Others_density',
                 'Others_AF'
                 ]

    corr_p = [spearmanr(dat[[i, stage]]) for i in tau_types]   # get both r, p
    r_val = [round(i[0], 3) for i in corr_p]  # separates r out
    p_val = [i[1] for i in corr_p]  # separates p out
    # [round(spearmanr(dat[[i, stage]])[0], 3) for i in tau_types]
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                stage: r_val,
                                'p_val': p_val
                                })
    return output


def correlation_region_tau_stage(dat, stage):
    """
    Calculates correlation between region-specific tau burden
    for each burden type & stage rating of choice e.g. Stage_SK.
    To see whcih region & tau burden type correlates best with Stage_SK.
    """
    kovacs_regions = ['Globus pallidus',
                      'Subthalamic nucleus',
                      'Striatum',
                      'Pre-frontal',
                      'Dentate nucleus',
                      'Occipital'
                      ]

    tau_types = ['Total_tau_density',
                 'Tau_hallmark_density',
                 'CB_density',
                 'NFT_density',
                 'TA_density',
                 'Others_density',
                 'Others_AF'
                ]
    regional_corr = []
    regional_pval = []
    # for each region, calculates corr with all tau types
    for r in kovacs_regions:
        r_dat = dat[dat['region_name'] == r]
        r_corr = [round(spearmanr(r_dat[[i, stage]])[0], 3) for i in tau_types]
        p_val = [round(spearmanr(r_dat[[i, stage]])[1], 3) for i in tau_types]
        regional_corr.append(r_corr)
        regional_pval.append(p_val)
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                kovacs_regions[0]: regional_corr[0],
                                kovacs_regions[0]+'_pval': regional_pval[0],
                                kovacs_regions[1]: regional_corr[1],
                                kovacs_regions[1]+'_pval': regional_pval[1],
                                kovacs_regions[2]: regional_corr[2],
                                kovacs_regions[2]+'_pval': regional_pval[2],
                                kovacs_regions[3]: regional_corr[3],
                                kovacs_regions[3]+'_pval': regional_pval[3],
                                kovacs_regions[4]: regional_corr[4],
                                kovacs_regions[4]+'_pval': regional_pval[4],
                                kovacs_regions[5]: regional_corr[5],
                                kovacs_regions[5]+'_pval': regional_pval[5]
                                })
    print('Correlation between region-specific tau & '+stage)
    return output


def correlation_region_tau_region_stage(dat):
    """
    Calculates correlation between region-specific tau burden
    for each burden type & region-specific severity rating.
    E.g. (tau in dentate nucleus & DE_SK)
    """
    kovacs_regions_rating = {'Globus pallidus': 'GP_SK',
                             'Subthalamic nucleus': 'STN_SK',
                             'Striatum': 'STR_SK',
                             'Pre-frontal': 'FCF_SK',
                             'Dentate nucleus': 'DE_SK',
                             'Occipital': 'OC_SK'
                             }

    tau_types = ['Total_tau_density',
                 'Tau_hallmark_density',
                 'CB_density',
                 'NFT_density',
                 'TA_density',
                 'Others_density',
                 'Others_AF'
                 ]

    regional_corr = []
    regional_pval = []
    kovacs_regions = list(kovacs_regions_rating.keys())

    # for each region, calculates corr with region-specific rating
    for r in kovacs_regions:
        r_dat = dat[dat['region_name'] == r]  # get region dat
        r_stage = kovacs_regions_rating[r]  # get region-specific rating
        r_dat_ = r_dat[~r_dat[r_stage].isna()]  # get rid of NAN
        r_corr = [round(spearmanr(r_dat_[[i,
                                          r_stage]])[0], 3) for i in tau_types]
        p_val = [round(spearmanr(r_dat_[[i,
                                          r_stage]])[1], 3) for i in tau_types]
        regional_corr.append(r_corr)
        regional_pval.append(p_val)
    output = pd.DataFrame(data={'Tau_burden': tau_types,
                                kovacs_regions[0]: regional_corr[0],
                                kovacs_regions[0]+'_pval': regional_pval[0],
                                kovacs_regions[1]: regional_corr[1],
                                kovacs_regions[1]+'_pval': regional_pval[1],
                                kovacs_regions[2]: regional_corr[2],
                                kovacs_regions[2]+'_pval': regional_pval[2],
                                kovacs_regions[3]: regional_corr[3],
                                kovacs_regions[3]+'_pval': regional_pval[3],
                                kovacs_regions[4]: regional_corr[4],
                                kovacs_regions[4]+'_pval': regional_pval[4],
                                kovacs_regions[5]: regional_corr[5],
                                kovacs_regions[5]+'_pval': regional_pval[5]
                                })
    print('Correlation between region-specific tau & region-specific rating')
    return output

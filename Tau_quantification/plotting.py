# helper functions for plotting data as polar plots

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# def melt_psp_by_stage(df, stage):
#     """
#     Subset original dataframe by psp stage & melt each subset dataframe
#     so that tau burden is in one column them and put in a dict.
#     """
#     stages = list(set(df[stage].astype(int)))
#     output_dict = {}
#     for s in stages:
#         stage_s = df[df[stage] == s]
#         stage_s_subset = stage_s[['Slice_ID',
#                                   'region_name',
#                                   'Tau_hallmark_density',
#                                   'NFT_density',
#                                   'TA_density',
#                                   'CB_density',
#                                   'Others_density',
#                                   'Others_AF'
#                                   ]]
#         stage_s_melt = stage_s_subset.melt(id_vars=['Slice_ID',
#                                                     'region_name'],
#                                            value_vars=['Tau_hallmark_density',
#                                                        'NFT_density',
#                                                        'TA_density',
#                                                        'CB_density',
#                                                        'Others_density',
#                                                        'Others_AF'
#                                                        ])
#         output_dict[s] = stage_s_melt
#     return output_dict


def subset_psp_stage(df, stage):
    """
    Subset original dataframe by psp stage & put in a dict.
    """
    stages = list(set(df[stage].astype(int)))
    output_dict = {}
    for s in stages:
        stage_s = df[df[stage] == s]
        stage_s_subset = stage_s[['Slice_ID',
                                  'region_name',
                                  'Total_tau_density',
                                  'Tau_hallmark_density',
                                  'NFT_density',
                                  'TA_density',
                                  'CB_density',
                                  'Others_density',
                                  'Others_AF'
                                  ]]
        output_dict[s] = stage_s_subset
    return output_dict


def mean_per_region(df, tau_type):
    """
    Calculates mean tau per region.
    """
    region_names = list(set(df['region_name']))
    r_means = []
    for r in region_names:
        r_mean = np.mean(df[df['region_name'] == r][tau_type])
        r_means.append(r_mean)
    output = pd.DataFrame(data={'region_name': region_names,
                                tau_type + '_mean': r_means})
    output = output.sort_values(by=tau_type + '_mean')
    return output


def mean_tau_stage(df_stage_dict, tau_type):
    """
    Taking a dictionary of subset dataframes by stage,
    and calculating mean tau_type for each subset dataframe.
    Return a dictionary out.
    """
    stage_mean = {}
    stages = list(df_stage_dict.keys())
    for s in stages:
        mean_roi = mean_per_region(df_stage_dict[s], tau_type)
        stage_mean[s] = mean_roi

    return stage_mean


def polar_plot(dict_subset,
               dict_mean,
               tau_type,
               val_add,
               anatomical_order,
               marker_colors,
               marker_color,
               marker_line_color,
               title,
               tick_label,
               fig_height,
               fig_width,
               tickangle_val):

    # get stage info
    stages = list(dict_subset.keys())
    min_stage = min(stages)
    max_stage = max(stages)

    # set plotting value range

    # If min is 0 , we take second lowest value, to avoid log10(0) = infinity

    if (min(dict_subset[min_stage][tau_type]) == 0):
        sorted_val = sorted(set(dict_subset[min_stage][tau_type]))[1]
    else:
        sorted_val = sorted(set(dict_subset[min_stage][tau_type]))[0]
    min_val = (np.log10(sorted_val))
    max_val = max(np.log10(dict_subset[max_stage][tau_type] + val_add))
    print('min value: ', min_val)
    print('max value: ', max_val)

    # Plotting
    fig = make_subplots(rows=3,
                        cols=2,
                        specs=[[{"type": "polar"} for _ in range(2)] for _ in range(3)],
                        subplot_titles=("Stage 2",
                                        "Stage 3",
                                        "Stage 4",
                                        "Stage 5",
                                        "Stage 6"
                                        )
                        )
    # set marker colors
    if (len(marker_colors) == 0):
        marker_colors = [marker_color]*len(stages)

    util_dict = {2: [marker_colors[0], 1, 1],  # color, row, column
                 3: [marker_colors[1], 1, 2],
                 4: [marker_colors[2], 2, 1],
                 5: [marker_colors[3], 2, 2],
                 6: [marker_colors[4], 3, 1]}

    for s in stages:
        fig.append_trace(
            go.Barpolar(
                r=dict_mean[s][tau_type+'_mean'],
                theta=dict_mean[s]['region_name'],
                marker=dict(color=util_dict[s][0]),
                marker_line_color=marker_line_color,
                name='Tau density, stage ' + str(s),
                showlegend=False
            ),
            row=util_dict[s][1], col=util_dict[s][2]
        )
        fig.append_trace(
            go.Scatterpolargl(
                r=dict_subset[s][tau_type],
                theta=dict_subset[s]['region_name'],
                name='Stage ' + str(s),
                mode='markers',
                marker_color=marker_line_color,
                marker=dict(size=4),
                showlegend=False
            ),
            row=util_dict[s][1], col=util_dict[s][2]
        )

    fig.update_layout(
                    height=fig_height,
                    width=fig_width,
                    title_text=title,
                    polar=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar2=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar3=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar4=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar5=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black'))
                    )

    fig.layout.annotations[0].update(x=0.025, font=dict(color='black'))
    fig.layout.annotations[1].update(x=0.575, font=dict(color='black'))
    fig.layout.annotations[2].update(x=0.025, font=dict(color='black'))
    fig.layout.annotations[3].update(x=0.575, font=dict(color='black'))
    fig.layout.annotations[4].update(x=0.025, font=dict(color='black'))
    return fig


def polar_plot_together(dict_subset,
                        list_dict_mean,
                        list_tau_type,
                        min_tau_type,
                        max_tau_type,
                        val_add,
                        anatomical_order,
                        tau_marker_colors,
                        marker_color,
                        marker_line_color,
                        title,
                        tick_label,
                        fig_height,
                        fig_width,
                        tickangle_val):

    # get stage info
    stages = list(dict_subset.keys())
    min_stage = min(stages)
    max_stage = max(stages)

    # set plotting value range
    val_array = np.array(dict_subset[min_stage][min_tau_type])
    min_val = min(np.log10(val_array[np.nonzero(val_array)]))
    max_val = max(np.log10(dict_subset[max_stage][max_tau_type] + val_add))
    print('min value: ', min_val)
    print('max value: ', max_val)

    # Plotting
    fig = make_subplots(rows=5,  # stage 2 - 6
                        cols=4,  # tau type: CB, NFT, Others, TA
                        specs=[[{"type": "polar"} for _ in range(4)] for _ in range(5)],
                        subplot_titles=("Stage 2", "NFT density", "TA density", "TF density ",
                                        "Stage 3", "CB density ", " ", " ",
                                        "Stage 4", " ", " ", " ",
                                        "Stage 5", " ", " ", " ",
                                        "Stage 6", " ", " ", " "
                                        )
                        # subplot_titles=('Stage 2: CB density', "NFT density", "Others density", "TA density",
                        #                 "Stage 3: CB density", "NFT density", "Others density", "TA density",
                        #                 "Stage 4: CB density", "NFT density", "Others density", "TA density",
                        #                 "Stage 5: CB density", "NFT density", "Others density", "TA density",
                        #                 "Stage 6: CB density", "NFT density", "Others density", "TA density"
                        #                 )
                        )
    # set marker colors
    if (len(tau_marker_colors) == 0):
        tau_marker_colors = [marker_color]*len(stages)

    util_dict = {2: [1, 1],  # row, column
                 3: [2, 1],
                 4: [3, 1],
                 5: [4, 1],
                 6: [5, 1]}

    tau_dict = {'CB_density': tau_marker_colors[0],
                'NFT_density': tau_marker_colors[1],
                'TA_density': tau_marker_colors[2],
                'Others_density': tau_marker_colors[3]
                }

    for i in range(0, len(list_dict_mean)):
        s = 2
        dict_mean = list_dict_mean[i]
        tau_type = list_tau_type[i]

        fig.append_trace(
            go.Barpolar(
                r=dict_mean[s][tau_type+'_mean'],
                theta=dict_mean[s]['region_name'],
                marker=dict(color=tau_dict[tau_type]),
                marker_line_color=marker_line_color,
                name=tau_type,
                showlegend=True
            ),
            row=util_dict[s][0], col=util_dict[s][1]+i
        )
        fig.append_trace(
            go.Scatterpolargl(
                r=dict_subset[s][tau_type],
                theta=dict_subset[s]['region_name'],
                name='Stage ' + str(s),
                mode='markers',
                marker_color=marker_line_color,
                marker=dict(size=4),
                showlegend=False
            ),
            row=util_dict[s][0], col=util_dict[s][1]+i
        )
    stages.pop(0)
    for s in stages:
        for i in range(0, len(list_dict_mean)):
            dict_mean = list_dict_mean[i]
            tau_type = list_tau_type[i]

            fig.append_trace(
                go.Barpolar(
                    r=dict_mean[s][tau_type+'_mean'],
                    theta=dict_mean[s]['region_name'],
                    marker=dict(color=tau_dict[tau_type]),
                    marker_line_color=marker_line_color,
                    name=tau_type+',s' + str(s),
                    showlegend=False
                ),
                row=util_dict[s][0], col=util_dict[s][1]+i
            )
            fig.append_trace(
                go.Scatterpolargl(
                    r=dict_subset[s][tau_type],
                    theta=dict_subset[s]['region_name'],
                    name='Stage ' + str(s),
                    mode='markers',
                    marker_color=marker_line_color,
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=util_dict[s][0], col=util_dict[s][1]+i
            )

    fig.update_layout(
                    height=fig_height,
                    width=fig_width,
                    title_text=title,
                    # stage 2
                    polar1=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar2=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar3=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar4=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    # stage 3
                    polar5=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar6=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar7=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar8=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    # stage 4
                    polar9=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size = 15),
                                         color='black')),
                    polar10=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar11=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar12=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    # stage 5
                    polar13=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar14=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar15=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar16=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    # stage 6
                    polar17=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar18=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar19=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black')),
                    polar20=dict(
                        radialaxis=dict(type='log',
                                        tickangle=tickangle_val,
                                        range=[min_val, max_val],
                                        showticklabels=tick_label,
                                        ticks='outside',
                                        dtick=1
                                        ),
                        angularaxis_categoryarray=anatomical_order,
                        angularaxis=dict(tickfont=dict(size=15),
                                         color='black'))
                    )
    # # stage 2
    fig.layout.annotations[0].update(y=1, x=0.10625-0.1,
                                      font=dict(color='black'))
    fig.layout.annotations[1].update(y=1+0.02,
                                     font=dict(color='black'))
    fig.layout.annotations[2].update(y=1+0.02,
                                     font=dict(color='black'))
    fig.layout.annotations[3].update(y=1+0.02,
                                     font=dict(color='black'))
    # # # stage 3
    fig.layout.annotations[4].update(y=0.78, x=0.10625-0.1,
                                      font=dict(color='black'))
    fig.layout.annotations[5].update(y=1+0.02, x=0.10625,
                                     font=dict(color='black'))
    # # fig.layout.annotations[6].update(y=0.78+0.02)
    # # fig.layout.annotations[7].update(y=0.78+0.02)
    # # # stage 4
    fig.layout.annotations[8].update(y=0.56, x=0.10625-0.1,
                                      font=dict(color='black'))
    # # fig.layout.annotations[9].update(y=0.56+0.02)
    # # fig.layout.annotations[10].update(y=0.56+0.02)
    # # fig.layout.annotations[11].update(y=0.56+0.02)
    # # # stage 5
    fig.layout.annotations[12].update(y=0.33999999999999997, x=0.10625-0.1,
                                      font=dict(color='black'))
    # # fig.layout.annotations[13].update(y=0.33999999999999997+0.02)
    # # fig.layout.annotations[14].update(y=0.33999999999999997+0.02)
    # # fig.layout.annotations[15].update(y=0.33999999999999997+0.02)
    # # # stage 6
    fig.layout.annotations[16].update(y=0.12, x=0.10625-0.1,
                                      font=dict(color='black'))
    # # fig.layout.annotations[17].update(y=0.12+0.02)
    # # fig.layout.annotations[18].update(y=0.12+0.02)
    # # fig.layout.annotations[19].update(y=0.12+0.02)

    return fig

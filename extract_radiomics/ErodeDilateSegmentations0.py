# -*- coding: utf-8 -*-
"""
Erode and dilate segmentations for radiomics feature ICCs evaluation.

Not for clinical use.
SPDX-FileCopyrightText: 2021 Medical Physics Unit, McGill University, Montreal, CAN
SPDX-FileCopyrightText: 2021 Thierry Lefebvre
SPDX-FileCopyrightText: 2021 Peter Savadjiev
SPDX-License-Identifier: MIT
"""

import SimpleITK as sitk
from os import listdir
from os.path import join

modality = 'MRI_SEQUENCE' # insert name of MRI sequence of interest (e.g. DCE2, ADC, DWI, etc.)

mypathseg      = 'MYPROJECTFILEPATH/SEG/'   +modality+'/'
mypathsave_dil = 'MYPROJECTFILEPATH/DILSEG/'+modality+'/'
mypathsave_ero = 'MYPROJECTFILEPATH/EROSEG/'+modality+'/'

dirsseg = listdir(mypathseg)
dirsseg.sort()

fullpathsseg = []


for dir2 in dirsseg:
    fullpathsseg.append(join(mypathseg,dir2))


for i in range(len(dirsseg)):
    print(dirsseg[i][0:-4])
    print(i)
    seg_img_init = sitk.ReadImage(fullpathsseg[i],sitk.sitkUInt8)
    
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius((1,1,1))
    dilate.SetForegroundValue(1)
    seg_img_dil = dilate.Execute(seg_img_init)
    seg_img_dil.SetSpacing(seg_img_init.GetSpacing())
    seg_img_dil.SetOrigin(seg_img_init.GetOrigin())
    sitk.WriteImage(seg_img_dil,join(mypathsave_dil,dirsseg[i]))
    
    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius((1,1,0))
    erode.SetForegroundValue(1)
    seg_img_ero = erode.Execute(seg_img_init)
    seg_img_ero.SetSpacing(seg_img_init.GetSpacing())
    seg_img_ero.SetOrigin(seg_img_init.GetOrigin())
    sitk.WriteImage(seg_img_ero,join(mypathsave_ero,dirsseg[i]))    


 
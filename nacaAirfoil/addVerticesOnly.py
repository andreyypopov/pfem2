#!/usr/bin/env python

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'C:/Users/AYPopov/Desktop/nacaAirfoil')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Vertex_0 = geompy.MakeVertex(-0.5, 0, 0)
Vertex_1 = geompy.MakeVertex(-0.499375, 0.00440616763538602, 0)
Vertex_2 = geompy.MakeVertex(-0.49875, 0.00620337080804088, 0)
Vertex_3 = geompy.MakeVertex(-0.4975, 0.00871668416288437, 0)
Vertex_4 = geompy.MakeVertex(-0.495, 0.0122131474837072, 0)
Vertex_5 = geompy.MakeVertex(-0.4925, 0.0148485818097754, 0)
Vertex_6 = geompy.MakeVertex(-0.49, 0.0170370739584, 0)
Vertex_7 = geompy.MakeVertex(-0.485, 0.0206367116996198, 0)
Vertex_8 = geompy.MakeVertex(-0.48, 0.0235977710945143, 0)
Vertex_9 = geompy.MakeVertex(-0.475, 0.0261471981503698, 0)
Vertex_10 = geompy.MakeVertex(-0.47, 0.028401444396432, 0)
Vertex_11 = geompy.MakeVertex(-0.46, 0.0322772219904, 0)
Vertex_12 = geompy.MakeVertex(-0.45, 0.0355468489511813, 0)
Vertex_13 = geompy.MakeVertex(-0.44, 0.0383757939643395, 0)
Vertex_14 = geompy.MakeVertex(-0.43, 0.0408627263337046, 0)
Vertex_15 = geompy.MakeVertex(-0.42, 0.0430722476866286, 0)
Vertex_16 = geompy.MakeVertex(-0.41, 0.0450494985024, 0)
Vertex_17 = geompy.MakeVertex(-0.4, 0.0468275782382395, 0)
Vertex_18 = geompy.MakeVertex(-0.39, 0.0484316791497911, 0)
Vertex_19 = geompy.MakeVertex(-0.38, 0.049881554914464, 0)
Vertex_20 = geompy.MakeVertex(-0.37, 0.0511930771635155, 0)
Vertex_21 = geompy.MakeVertex(-0.36, 0.052379260822391, 0)
Vertex_22 = geompy.MakeVertex(-0.35, 0.0534509643293389, 0)
Vertex_23 = geompy.MakeVertex(-0.34, 0.0544173825024, 0)
Vertex_24 = geompy.MakeVertex(-0.33, 0.055286402501153, 0)
Vertex_25 = geompy.MakeVertex(-0.32, 0.0560648666787429, 0)
Vertex_26 = geompy.MakeVertex(-0.31, 0.0567587704666336, 0)
Vertex_27 = geompy.MakeVertex(-0.3, 0.0573734139023625, 0)
Vertex_28 = geompy.MakeVertex(-0.29, 0.0579135194203433, 0)
Vertex_29 = geompy.MakeVertex(-0.28, 0.0583833246558946, 0)
Vertex_30 = geompy.MakeVertex(-0.27, 0.0587866564506928, 0)
Vertex_31 = geompy.MakeVertex(-0.26, 0.0591269905142791, 0)
Vertex_32 = geompy.MakeVertex(-0.25, 0.0594075, 0)
Vertex_33 = geompy.MakeVertex(-0.24, 0.0596310954135419, 0)
Vertex_34 = geompy.MakeVertex(-0.23, 0.0598004576724959, 0)
Vertex_35 = geompy.MakeVertex(-0.22, 0.0599180657010092, 0)
Vertex_36 = geompy.MakeVertex(-0.21, 0.0599862196246941, 0)
Vertex_37 = geompy.MakeVertex(-0.2, 0.0600070603939703, 0)
Vertex_38 = geompy.MakeVertex(-0.19, 0.059982586485854, 0)
Vertex_39 = geompy.MakeVertex(-0.18, 0.0599146681988573, 0)
Vertex_40 = geompy.MakeVertex(-0.17, 0.0598050599518284, 0)
Vertex_41 = geompy.MakeVertex(-0.16, 0.0596554109171742, 0)
Vertex_42 = geompy.MakeVertex(-0.15, 0.0594672742561365, 0)
Vertex_43 = geompy.MakeVertex(-0.14, 0.0592421151744, 0)
Vertex_44 = geompy.MakeVertex(-0.13, 0.0589813179771325, 0)
Vertex_45 = geompy.MakeVertex(-0.12, 0.0586861922712893, 0)
Vertex_46 = geompy.MakeVertex(-0.11, 0.0583579784378691, 0)
Vertex_47 = geompy.MakeVertex(-0.1, 0.057997852476479, 0)
Vertex_48 = geompy.MakeVertex(-0.09, 0.0576069303080288, 0)
Vertex_49 = geompy.MakeVertex(-0.08, 0.0571862716078376, 0)
Vertex_50 = geompy.MakeVertex(-0.07, 0.0567368832303158, 0)
Vertex_51 = geompy.MakeVertex(-0.06, 0.0562597222771822, 0)
Vertex_52 = geompy.MakeVertex(-0.05, 0.0557556988535438, 0)
Vertex_53 = geompy.MakeVertex(-0.04, 0.0552256785497935, 0)
Vertex_54 = geompy.MakeVertex(-0.03, 0.0546704846819442, 0)
Vertex_55 = geompy.MakeVertex(-0.02, 0.0540909003185279, 0)
Vertex_56 = geompy.MakeVertex(-0.01, 0.0534876701184, 0)
Vertex_57 = geompy.MakeVertex(0, 0.0528615020005716, 0)
Vertex_58 = geompy.MakeVertex(0.01, 0.0522130686644623, 0)
Vertex_59 = geompy.MakeVertex(0.02, 0.051543008976631, 0)
Vertex_60 = geompy.MakeVertex(0.03, 0.0508519292380431, 0)
Vertex_61 = geompy.MakeVertex(0.04, 0.0501404043442186, 0)
Vertex_62 = geompy.MakeVertex(0.05, 0.0494089788491221, 0)
Vertex_63 = geompy.MakeVertex(0.0600000000000001, 0.048658167942382, 0)
Vertex_64 = geompy.MakeVertex(0.07, 0.0478884583483131, 0)
Vertex_65 = geompy.MakeVertex(0.08, 0.0471003091542597, 0)
Vertex_66 = geompy.MakeVertex(0.09, 0.0462941525749314, 0)
Vertex_67 = geompy.MakeVertex(0.1, 0.0454703946586779, 0)
Vertex_68 = geompy.MakeVertex(0.11, 0.0446294159410011, 0)
Vertex_69 = geompy.MakeVertex(0.12, 0.0437715720500464, 0)
Vertex_70 = geompy.MakeVertex(0.13, 0.0428971942683138, 0)
Vertex_71 = geompy.MakeVertex(0.14, 0.0420065900544, 0)
Vertex_72 = geompy.MakeVertex(0.15, 0.0411000435281904, 0)
Vertex_73 = geompy.MakeVertex(0.16, 0.040177815922585, 0)
Vertex_74 = geompy.MakeVertex(0.17, 0.0392401460045358, 0)
Vertex_75 = geompy.MakeVertex(0.18, 0.038287250467906, 0)
Vertex_76 = geompy.MakeVertex(0.19, 0.0373193243004226, 0)
Vertex_77 = geompy.MakeVertex(0.2, 0.0363365411267802, 0)
Vertex_78 = geompy.MakeVertex(0.21, 0.0353390535297636, 0)
Vertex_79 = geompy.MakeVertex(0.22, 0.0343269933510859, 0)
Vertex_80 = geompy.MakeVertex(0.23, 0.0333004719734865, 0)
Vertex_81 = geompy.MakeVertex(0.24, 0.0322595805854973, 0)
Vertex_82 = geompy.MakeVertex(0.25, 0.0312043904301599, 0)
Vertex_83 = geompy.MakeVertex(0.26, 0.0301349530388671, 0)
Vertex_84 = geompy.MakeVertex(0.27, 0.0290513004514033, 0)
Vertex_85 = geompy.MakeVertex(0.28, 0.0279534454231642, 0)
Vertex_86 = geompy.MakeVertex(0.29, 0.0268413816204599, 0)
Vertex_87 = geompy.MakeVertex(0.3, 0.025715083804725, 0)
Vertex_88 = geompy.MakeVertex(0.31, 0.0245745080064, 0)
Vertex_89 = geompy.MakeVertex(0.32, 0.0234195916891799, 0)
Vertex_90 = geompy.MakeVertex(0.33, 0.0222502539052765, 0)
Vertex_91 = geompy.MakeVertex(0.34, 0.0210663954422867, 0)
Vertex_92 = geompy.MakeVertex(0.35, 0.0198678989622155, 0)
Vertex_93 = geompy.MakeVertex(0.36, 0.0186546291331604, 0)
Vertex_94 = geompy.MakeVertex(0.37, 0.0174264327541241, 0)
Vertex_95 = geompy.MakeVertex(0.38, 0.0161831388733891, 0)
Vertex_96 = geompy.MakeVertex(0.39, 0.0149245589008563, 0)
Vertex_97 = geompy.MakeVertex(0.4, 0.0136504867147185, 0)
Vertex_98 = geompy.MakeVertex(0.41, 0.0123606987628147, 0)
Vertex_99 = geompy.MakeVertex(0.42, 0.0110549541589856, 0)
Vertex_100 = geompy.MakeVertex(0.43, 0.00973299477472849, 0)
Vertex_101 = geompy.MakeVertex(0.44, 0.00839454532642893, 0)
Vertex_102 = geompy.MakeVertex(0.45, 0.00703931345842686, 0)
Vertex_103 = geompy.MakeVertex(0.46, 0.0056669898221581, 0)
Vertex_104 = geompy.MakeVertex(0.47, 0.0042772481515958, 0)
Vertex_105 = geompy.MakeVertex(0.48, 0.00286974533520023, 0)
Vertex_106 = geompy.MakeVertex(0.49, 0.00144412148457324, 0)
Vertex_107 = geompy.MakeVertex(0.5, -1.66533453693773E-17, 0)
Vertex_108 = geompy.MakeVertex(-0.499375, -0.00440616763538602, 0)
Vertex_109 = geompy.MakeVertex(-0.49875, -0.00620337080804088, 0)
Vertex_110 = geompy.MakeVertex(-0.4975, -0.00871668416288437, 0)
Vertex_111 = geompy.MakeVertex(-0.495, -0.0122131474837072, 0)
Vertex_112 = geompy.MakeVertex(-0.4925, -0.0148485818097754, 0)
Vertex_113 = geompy.MakeVertex(-0.49, -0.0170370739584, 0)
Vertex_114 = geompy.MakeVertex(-0.485, -0.0206367116996198, 0)
Vertex_115 = geompy.MakeVertex(-0.48, -0.0235977710945143, 0)
Vertex_116 = geompy.MakeVertex(-0.475, -0.0261471981503698, 0)
Vertex_117 = geompy.MakeVertex(-0.47, -0.028401444396432, 0)
Vertex_118 = geompy.MakeVertex(-0.46, -0.0322772219904, 0)
Vertex_119 = geompy.MakeVertex(-0.45, -0.0355468489511813, 0)
Vertex_120 = geompy.MakeVertex(-0.44, -0.0383757939643395, 0)
Vertex_121 = geompy.MakeVertex(-0.43, -0.0408627263337046, 0)
Vertex_122 = geompy.MakeVertex(-0.42, -0.0430722476866286, 0)
Vertex_123 = geompy.MakeVertex(-0.41, -0.0450494985024, 0)
Vertex_124 = geompy.MakeVertex(-0.4, -0.0468275782382395, 0)
Vertex_125 = geompy.MakeVertex(-0.39, -0.0484316791497911, 0)
Vertex_126 = geompy.MakeVertex(-0.38, -0.049881554914464, 0)
Vertex_127 = geompy.MakeVertex(-0.37, -0.0511930771635155, 0)
Vertex_128 = geompy.MakeVertex(-0.36, -0.052379260822391, 0)
Vertex_129 = geompy.MakeVertex(-0.35, -0.0534509643293389, 0)
Vertex_130 = geompy.MakeVertex(-0.34, -0.0544173825024, 0)
Vertex_131 = geompy.MakeVertex(-0.33, -0.055286402501153, 0)
Vertex_132 = geompy.MakeVertex(-0.32, -0.0560648666787429, 0)
Vertex_133 = geompy.MakeVertex(-0.31, -0.0567587704666336, 0)
Vertex_134 = geompy.MakeVertex(-0.3, -0.0573734139023625, 0)
Vertex_135 = geompy.MakeVertex(-0.29, -0.0579135194203433, 0)
Vertex_136 = geompy.MakeVertex(-0.28, -0.0583833246558946, 0)
Vertex_137 = geompy.MakeVertex(-0.27, -0.0587866564506928, 0)
Vertex_138 = geompy.MakeVertex(-0.26, -0.0591269905142791, 0)
Vertex_139 = geompy.MakeVertex(-0.25, -0.0594075, 0)
Vertex_140 = geompy.MakeVertex(-0.24, -0.0596310954135419, 0)
Vertex_141 = geompy.MakeVertex(-0.23, -0.0598004576724959, 0)
Vertex_142 = geompy.MakeVertex(-0.22, -0.0599180657010092, 0)
Vertex_143 = geompy.MakeVertex(-0.21, -0.0599862196246941, 0)
Vertex_144 = geompy.MakeVertex(-0.2, -0.0600070603939703, 0)
Vertex_145 = geompy.MakeVertex(-0.19, -0.059982586485854, 0)
Vertex_146 = geompy.MakeVertex(-0.18, -0.0599146681988573, 0)
Vertex_147 = geompy.MakeVertex(-0.17, -0.0598050599518284, 0)
Vertex_148 = geompy.MakeVertex(-0.16, -0.0596554109171742, 0)
Vertex_149 = geompy.MakeVertex(-0.15, -0.0594672742561365, 0)
Vertex_150 = geompy.MakeVertex(-0.14, -0.0592421151744, 0)
Vertex_151 = geompy.MakeVertex(-0.13, -0.0589813179771325, 0)
Vertex_152 = geompy.MakeVertex(-0.12, -0.0586861922712893, 0)
Vertex_153 = geompy.MakeVertex(-0.11, -0.0583579784378691, 0)
Vertex_154 = geompy.MakeVertex(-0.1, -0.057997852476479, 0)
Vertex_155 = geompy.MakeVertex(-0.09, -0.0576069303080288, 0)
Vertex_156 = geompy.MakeVertex(-0.08, -0.0571862716078376, 0)
Vertex_157 = geompy.MakeVertex(-0.07, -0.0567368832303158, 0)
Vertex_158 = geompy.MakeVertex(-0.06, -0.0562597222771822, 0)
Vertex_159 = geompy.MakeVertex(-0.05, -0.0557556988535438, 0)
Vertex_160 = geompy.MakeVertex(-0.04, -0.0552256785497935, 0)
Vertex_161 = geompy.MakeVertex(-0.03, -0.0546704846819442, 0)
Vertex_162 = geompy.MakeVertex(-0.02, -0.0540909003185279, 0)
Vertex_163 = geompy.MakeVertex(-0.01, -0.0534876701184, 0)
Vertex_164 = geompy.MakeVertex(0, -0.0528615020005716, 0)
Vertex_165 = geompy.MakeVertex(0.01, -0.0522130686644623, 0)
Vertex_166 = geompy.MakeVertex(0.02, -0.051543008976631, 0)
Vertex_167 = geompy.MakeVertex(0.03, -0.0508519292380431, 0)
Vertex_168 = geompy.MakeVertex(0.04, -0.0501404043442186, 0)
Vertex_169 = geompy.MakeVertex(0.05, -0.0494089788491221, 0)
Vertex_170 = geompy.MakeVertex(0.0600000000000001, -0.048658167942382, 0)
Vertex_171 = geompy.MakeVertex(0.07, -0.0478884583483131, 0)
Vertex_172 = geompy.MakeVertex(0.08, -0.0471003091542597, 0)
Vertex_173 = geompy.MakeVertex(0.09, -0.0462941525749314, 0)
Vertex_174 = geompy.MakeVertex(0.1, -0.0454703946586779, 0)
Vertex_175 = geompy.MakeVertex(0.11, -0.0446294159410011, 0)
Vertex_176 = geompy.MakeVertex(0.12, -0.0437715720500464, 0)
Vertex_177 = geompy.MakeVertex(0.13, -0.0428971942683138, 0)
Vertex_178 = geompy.MakeVertex(0.14, -0.0420065900544, 0)
Vertex_179 = geompy.MakeVertex(0.15, -0.0411000435281904, 0)
Vertex_180 = geompy.MakeVertex(0.16, -0.040177815922585, 0)
Vertex_181 = geompy.MakeVertex(0.17, -0.0392401460045358, 0)
Vertex_182 = geompy.MakeVertex(0.18, -0.038287250467906, 0)
Vertex_183 = geompy.MakeVertex(0.19, -0.0373193243004226, 0)
Vertex_184 = geompy.MakeVertex(0.2, -0.0363365411267802, 0)
Vertex_185 = geompy.MakeVertex(0.21, -0.0353390535297636, 0)
Vertex_186 = geompy.MakeVertex(0.22, -0.0343269933510859, 0)
Vertex_187 = geompy.MakeVertex(0.23, -0.0333004719734865, 0)
Vertex_188 = geompy.MakeVertex(0.24, -0.0322595805854973, 0)
Vertex_189 = geompy.MakeVertex(0.25, -0.0312043904301599, 0)
Vertex_190 = geompy.MakeVertex(0.26, -0.0301349530388671, 0)
Vertex_191 = geompy.MakeVertex(0.27, -0.0290513004514033, 0)
Vertex_192 = geompy.MakeVertex(0.28, -0.0279534454231642, 0)
Vertex_193 = geompy.MakeVertex(0.29, -0.0268413816204599, 0)
Vertex_194 = geompy.MakeVertex(0.3, -0.025715083804725, 0)
Vertex_195 = geompy.MakeVertex(0.31, -0.0245745080064, 0)
Vertex_196 = geompy.MakeVertex(0.32, -0.0234195916891799, 0)
Vertex_197 = geompy.MakeVertex(0.33, -0.0222502539052765, 0)
Vertex_198 = geompy.MakeVertex(0.34, -0.0210663954422867, 0)
Vertex_199 = geompy.MakeVertex(0.35, -0.0198678989622155, 0)
Vertex_200 = geompy.MakeVertex(0.36, -0.0186546291331604, 0)
Vertex_201 = geompy.MakeVertex(0.37, -0.0174264327541241, 0)
Vertex_202 = geompy.MakeVertex(0.38, -0.0161831388733891, 0)
Vertex_203 = geompy.MakeVertex(0.39, -0.0149245589008563, 0)
Vertex_204 = geompy.MakeVertex(0.4, -0.0136504867147185, 0)
Vertex_205 = geompy.MakeVertex(0.41, -0.0123606987628147, 0)
Vertex_206 = geompy.MakeVertex(0.42, -0.0110549541589856, 0)
Vertex_207 = geompy.MakeVertex(0.43, -0.00973299477472849, 0)
Vertex_208 = geompy.MakeVertex(0.44, -0.00839454532642893, 0)
Vertex_209 = geompy.MakeVertex(0.45, -0.00703931345842686, 0)
Vertex_210 = geompy.MakeVertex(0.46, -0.0056669898221581, 0)
Vertex_211 = geompy.MakeVertex(0.47, -0.0042772481515958, 0)
Vertex_212 = geompy.MakeVertex(0.48, -0.00286974533520023, 0)
Vertex_213 = geompy.MakeVertex(0.49, -0.00144412148457324, 0)
Vertex_214 = geompy.MakeVertex(0.5, 1.66533453693773E-17, 0)

geomObj_1 = geompy.MakeMarker(0, 0, 0, 1, 0, 0, 0, 1, 0)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Vertex_0, 'Vertex_0' )
geompy.addToStudy( Vertex_1, 'Vertex_1' )
geompy.addToStudy( Vertex_2, 'Vertex_2' )
geompy.addToStudy( Vertex_3, 'Vertex_3' )
geompy.addToStudy( Vertex_4, 'Vertex_4' )
geompy.addToStudy( Vertex_5, 'Vertex_5' )
geompy.addToStudy( Vertex_6, 'Vertex_6' )
geompy.addToStudy( Vertex_7, 'Vertex_7' )
geompy.addToStudy( Vertex_8, 'Vertex_8' )
geompy.addToStudy( Vertex_9, 'Vertex_9' )
geompy.addToStudy( Vertex_10, 'Vertex_10' )
geompy.addToStudy( Vertex_11, 'Vertex_11' )
geompy.addToStudy( Vertex_12, 'Vertex_12' )
geompy.addToStudy( Vertex_13, 'Vertex_13' )
geompy.addToStudy( Vertex_14, 'Vertex_14' )
geompy.addToStudy( Vertex_15, 'Vertex_15' )
geompy.addToStudy( Vertex_16, 'Vertex_16' )
geompy.addToStudy( Vertex_17, 'Vertex_17' )
geompy.addToStudy( Vertex_18, 'Vertex_18' )
geompy.addToStudy( Vertex_19, 'Vertex_19' )
geompy.addToStudy( Vertex_20, 'Vertex_20' )
geompy.addToStudy( Vertex_21, 'Vertex_21' )
geompy.addToStudy( Vertex_22, 'Vertex_22' )
geompy.addToStudy( Vertex_23, 'Vertex_23' )
geompy.addToStudy( Vertex_24, 'Vertex_24' )
geompy.addToStudy( Vertex_25, 'Vertex_25' )
geompy.addToStudy( Vertex_26, 'Vertex_26' )
geompy.addToStudy( Vertex_27, 'Vertex_27' )
geompy.addToStudy( Vertex_28, 'Vertex_28' )
geompy.addToStudy( Vertex_29, 'Vertex_29' )
geompy.addToStudy( Vertex_30, 'Vertex_30' )
geompy.addToStudy( Vertex_31, 'Vertex_31' )
geompy.addToStudy( Vertex_32, 'Vertex_32' )
geompy.addToStudy( Vertex_33, 'Vertex_33' )
geompy.addToStudy( Vertex_34, 'Vertex_34' )
geompy.addToStudy( Vertex_35, 'Vertex_35' )
geompy.addToStudy( Vertex_36, 'Vertex_36' )
geompy.addToStudy( Vertex_37, 'Vertex_37' )
geompy.addToStudy( Vertex_38, 'Vertex_38' )
geompy.addToStudy( Vertex_39, 'Vertex_39' )
geompy.addToStudy( Vertex_40, 'Vertex_40' )
geompy.addToStudy( Vertex_41, 'Vertex_41' )
geompy.addToStudy( Vertex_42, 'Vertex_42' )
geompy.addToStudy( Vertex_43, 'Vertex_43' )
geompy.addToStudy( Vertex_44, 'Vertex_44' )
geompy.addToStudy( Vertex_45, 'Vertex_45' )
geompy.addToStudy( Vertex_46, 'Vertex_46' )
geompy.addToStudy( Vertex_47, 'Vertex_47' )
geompy.addToStudy( Vertex_48, 'Vertex_48' )
geompy.addToStudy( Vertex_49, 'Vertex_49' )
geompy.addToStudy( Vertex_50, 'Vertex_50' )
geompy.addToStudy( Vertex_51, 'Vertex_51' )
geompy.addToStudy( Vertex_52, 'Vertex_52' )
geompy.addToStudy( Vertex_53, 'Vertex_53' )
geompy.addToStudy( Vertex_54, 'Vertex_54' )
geompy.addToStudy( Vertex_55, 'Vertex_55' )
geompy.addToStudy( Vertex_56, 'Vertex_56' )
geompy.addToStudy( Vertex_57, 'Vertex_57' )
geompy.addToStudy( Vertex_58, 'Vertex_58' )
geompy.addToStudy( Vertex_59, 'Vertex_59' )
geompy.addToStudy( Vertex_60, 'Vertex_60' )
geompy.addToStudy( Vertex_61, 'Vertex_61' )
geompy.addToStudy( Vertex_62, 'Vertex_62' )
geompy.addToStudy( Vertex_63, 'Vertex_63' )
geompy.addToStudy( Vertex_64, 'Vertex_64' )
geompy.addToStudy( Vertex_65, 'Vertex_65' )
geompy.addToStudy( Vertex_66, 'Vertex_66' )
geompy.addToStudy( Vertex_67, 'Vertex_67' )
geompy.addToStudy( Vertex_68, 'Vertex_68' )
geompy.addToStudy( Vertex_69, 'Vertex_69' )
geompy.addToStudy( Vertex_70, 'Vertex_70' )
geompy.addToStudy( Vertex_71, 'Vertex_71' )
geompy.addToStudy( Vertex_72, 'Vertex_72' )
geompy.addToStudy( Vertex_73, 'Vertex_73' )
geompy.addToStudy( Vertex_74, 'Vertex_74' )
geompy.addToStudy( Vertex_75, 'Vertex_75' )
geompy.addToStudy( Vertex_76, 'Vertex_76' )
geompy.addToStudy( Vertex_77, 'Vertex_77' )
geompy.addToStudy( Vertex_78, 'Vertex_78' )
geompy.addToStudy( Vertex_79, 'Vertex_79' )
geompy.addToStudy( Vertex_80, 'Vertex_80' )
geompy.addToStudy( Vertex_81, 'Vertex_81' )
geompy.addToStudy( Vertex_82, 'Vertex_82' )
geompy.addToStudy( Vertex_83, 'Vertex_83' )
geompy.addToStudy( Vertex_84, 'Vertex_84' )
geompy.addToStudy( Vertex_85, 'Vertex_85' )
geompy.addToStudy( Vertex_86, 'Vertex_86' )
geompy.addToStudy( Vertex_87, 'Vertex_87' )
geompy.addToStudy( Vertex_88, 'Vertex_88' )
geompy.addToStudy( Vertex_89, 'Vertex_89' )
geompy.addToStudy( Vertex_90, 'Vertex_90' )
geompy.addToStudy( Vertex_91, 'Vertex_91' )
geompy.addToStudy( Vertex_92, 'Vertex_92' )
geompy.addToStudy( Vertex_93, 'Vertex_93' )
geompy.addToStudy( Vertex_94, 'Vertex_94' )
geompy.addToStudy( Vertex_95, 'Vertex_95' )
geompy.addToStudy( Vertex_96, 'Vertex_96' )
geompy.addToStudy( Vertex_97, 'Vertex_97' )
geompy.addToStudy( Vertex_98, 'Vertex_98' )
geompy.addToStudy( Vertex_99, 'Vertex_99' )
geompy.addToStudy( Vertex_100, 'Vertex_100' )
geompy.addToStudy( Vertex_101, 'Vertex_101' )
geompy.addToStudy( Vertex_102, 'Vertex_102' )
geompy.addToStudy( Vertex_103, 'Vertex_103' )
geompy.addToStudy( Vertex_104, 'Vertex_104' )
geompy.addToStudy( Vertex_105, 'Vertex_105' )
geompy.addToStudy( Vertex_106, 'Vertex_106' )
geompy.addToStudy( Vertex_107, 'Vertex_107' )
geompy.addToStudy( Vertex_108, 'Vertex_108' )
geompy.addToStudy( Vertex_109, 'Vertex_109' )
geompy.addToStudy( Vertex_110, 'Vertex_110' )
geompy.addToStudy( Vertex_111, 'Vertex_111' )
geompy.addToStudy( Vertex_112, 'Vertex_112' )
geompy.addToStudy( Vertex_113, 'Vertex_113' )
geompy.addToStudy( Vertex_114, 'Vertex_114' )
geompy.addToStudy( Vertex_115, 'Vertex_115' )
geompy.addToStudy( Vertex_116, 'Vertex_116' )
geompy.addToStudy( Vertex_117, 'Vertex_117' )
geompy.addToStudy( Vertex_118, 'Vertex_118' )
geompy.addToStudy( Vertex_119, 'Vertex_119' )
geompy.addToStudy( Vertex_120, 'Vertex_120' )
geompy.addToStudy( Vertex_121, 'Vertex_121' )
geompy.addToStudy( Vertex_122, 'Vertex_122' )
geompy.addToStudy( Vertex_123, 'Vertex_123' )
geompy.addToStudy( Vertex_124, 'Vertex_124' )
geompy.addToStudy( Vertex_125, 'Vertex_125' )
geompy.addToStudy( Vertex_126, 'Vertex_126' )
geompy.addToStudy( Vertex_127, 'Vertex_127' )
geompy.addToStudy( Vertex_128, 'Vertex_128' )
geompy.addToStudy( Vertex_129, 'Vertex_129' )
geompy.addToStudy( Vertex_130, 'Vertex_130' )
geompy.addToStudy( Vertex_131, 'Vertex_131' )
geompy.addToStudy( Vertex_132, 'Vertex_132' )
geompy.addToStudy( Vertex_133, 'Vertex_133' )
geompy.addToStudy( Vertex_134, 'Vertex_134' )
geompy.addToStudy( Vertex_135, 'Vertex_135' )
geompy.addToStudy( Vertex_136, 'Vertex_136' )
geompy.addToStudy( Vertex_137, 'Vertex_137' )
geompy.addToStudy( Vertex_138, 'Vertex_138' )
geompy.addToStudy( Vertex_139, 'Vertex_139' )
geompy.addToStudy( Vertex_140, 'Vertex_140' )
geompy.addToStudy( Vertex_141, 'Vertex_141' )
geompy.addToStudy( Vertex_142, 'Vertex_142' )
geompy.addToStudy( Vertex_143, 'Vertex_143' )
geompy.addToStudy( Vertex_144, 'Vertex_144' )
geompy.addToStudy( Vertex_145, 'Vertex_145' )
geompy.addToStudy( Vertex_146, 'Vertex_146' )
geompy.addToStudy( Vertex_147, 'Vertex_147' )
geompy.addToStudy( Vertex_148, 'Vertex_148' )
geompy.addToStudy( Vertex_149, 'Vertex_149' )
geompy.addToStudy( Vertex_150, 'Vertex_150' )
geompy.addToStudy( Vertex_151, 'Vertex_151' )
geompy.addToStudy( Vertex_152, 'Vertex_152' )
geompy.addToStudy( Vertex_153, 'Vertex_153' )
geompy.addToStudy( Vertex_154, 'Vertex_154' )
geompy.addToStudy( Vertex_155, 'Vertex_155' )
geompy.addToStudy( Vertex_156, 'Vertex_156' )
geompy.addToStudy( Vertex_157, 'Vertex_157' )
geompy.addToStudy( Vertex_158, 'Vertex_158' )
geompy.addToStudy( Vertex_159, 'Vertex_159' )
geompy.addToStudy( Vertex_160, 'Vertex_160' )
geompy.addToStudy( Vertex_161, 'Vertex_161' )
geompy.addToStudy( Vertex_162, 'Vertex_162' )
geompy.addToStudy( Vertex_163, 'Vertex_163' )
geompy.addToStudy( Vertex_164, 'Vertex_164' )
geompy.addToStudy( Vertex_165, 'Vertex_165' )
geompy.addToStudy( Vertex_166, 'Vertex_166' )
geompy.addToStudy( Vertex_167, 'Vertex_167' )
geompy.addToStudy( Vertex_168, 'Vertex_168' )
geompy.addToStudy( Vertex_169, 'Vertex_169' )
geompy.addToStudy( Vertex_170, 'Vertex_170' )
geompy.addToStudy( Vertex_171, 'Vertex_171' )
geompy.addToStudy( Vertex_172, 'Vertex_172' )
geompy.addToStudy( Vertex_173, 'Vertex_173' )
geompy.addToStudy( Vertex_174, 'Vertex_174' )
geompy.addToStudy( Vertex_175, 'Vertex_175' )
geompy.addToStudy( Vertex_176, 'Vertex_176' )
geompy.addToStudy( Vertex_177, 'Vertex_177' )
geompy.addToStudy( Vertex_178, 'Vertex_178' )
geompy.addToStudy( Vertex_179, 'Vertex_179' )
geompy.addToStudy( Vertex_180, 'Vertex_180' )
geompy.addToStudy( Vertex_181, 'Vertex_181' )
geompy.addToStudy( Vertex_182, 'Vertex_182' )
geompy.addToStudy( Vertex_183, 'Vertex_183' )
geompy.addToStudy( Vertex_184, 'Vertex_184' )
geompy.addToStudy( Vertex_185, 'Vertex_185' )
geompy.addToStudy( Vertex_186, 'Vertex_186' )
geompy.addToStudy( Vertex_187, 'Vertex_187' )
geompy.addToStudy( Vertex_188, 'Vertex_188' )
geompy.addToStudy( Vertex_189, 'Vertex_189' )
geompy.addToStudy( Vertex_190, 'Vertex_190' )
geompy.addToStudy( Vertex_191, 'Vertex_191' )
geompy.addToStudy( Vertex_192, 'Vertex_192' )
geompy.addToStudy( Vertex_193, 'Vertex_193' )
geompy.addToStudy( Vertex_194, 'Vertex_194' )
geompy.addToStudy( Vertex_195, 'Vertex_195' )
geompy.addToStudy( Vertex_196, 'Vertex_196' )
geompy.addToStudy( Vertex_197, 'Vertex_197' )
geompy.addToStudy( Vertex_198, 'Vertex_198' )
geompy.addToStudy( Vertex_199, 'Vertex_199' )
geompy.addToStudy( Vertex_200, 'Vertex_200' )
geompy.addToStudy( Vertex_201, 'Vertex_201' )
geompy.addToStudy( Vertex_202, 'Vertex_202' )
geompy.addToStudy( Vertex_203, 'Vertex_203' )
geompy.addToStudy( Vertex_204, 'Vertex_204' )
geompy.addToStudy( Vertex_205, 'Vertex_205' )
geompy.addToStudy( Vertex_206, 'Vertex_206' )
geompy.addToStudy( Vertex_207, 'Vertex_207' )
geompy.addToStudy( Vertex_208, 'Vertex_208' )
geompy.addToStudy( Vertex_209, 'Vertex_209' )
geompy.addToStudy( Vertex_210, 'Vertex_210' )
geompy.addToStudy( Vertex_211, 'Vertex_211' )
geompy.addToStudy( Vertex_212, 'Vertex_212' )
geompy.addToStudy( Vertex_213, 'Vertex_213' )
geompy.addToStudy( Vertex_214, 'Vertex_214' )

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()

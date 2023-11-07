{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f75344c5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:43.344315Z",
     "iopub.status.busy": "2022-01-26T14:19:43.343625Z",
     "iopub.status.idle": "2022-01-26T14:19:44.396797Z",
     "shell.execute_reply": "2022-01-26T14:19:44.396055Z",
     "shell.execute_reply.started": "2022-01-26T12:46:03.692767Z"
    },
    "papermill": {
     "duration": 1.112674,
     "end_time": "2022-01-26T14:19:44.396964",
     "exception": false,
     "start_time": "2022-01-26T14:19:43.284290",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dependencies Imported\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "print('Dependencies Imported')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e3536ee1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:44.502712Z",
     "iopub.status.busy": "2022-01-26T14:19:44.501653Z",
     "iopub.status.idle": "2022-01-26T14:19:44.700849Z",
     "shell.execute_reply": "2022-01-26T14:19:44.701361Z",
     "shell.execute_reply.started": "2022-01-26T12:46:47.573105Z"
    },
    "papermill": {
     "duration": 0.253163,
     "end_time": "2022-01-26T14:19:44.701566",
     "exception": false,
     "start_time": "2022-01-26T14:19:44.448403",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Course Name</th>\n",
       "      <th>University</th>\n",
       "      <th>Difficulty Level</th>\n",
       "      <th>Course Rating</th>\n",
       "      <th>Course URL</th>\n",
       "      <th>Course Description</th>\n",
       "      <th>Skills</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write A Feature Length Screenplay For Film Or ...</td>\n",
       "      <td>Michigan State University</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>4.8</td>\n",
       "      <td>https://www.coursera.org/learn/write-a-feature...</td>\n",
       "      <td>Write a Full Length Feature Film Script  In th...</td>\n",
       "      <td>Drama  Comedy  peering  screenwriting  film  D...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business Strategy: Business Model Canvas Analy...</td>\n",
       "      <td>Coursera Project Network</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>4.8</td>\n",
       "      <td>https://www.coursera.org/learn/canvas-analysis...</td>\n",
       "      <td>By the end of this guided project, you will be...</td>\n",
       "      <td>Finance  business plan  persona (user experien...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon Thin Film Solar Cells</td>\n",
       "      <td>�cole Polytechnique</td>\n",
       "      <td>Advanced</td>\n",
       "      <td>4.1</td>\n",
       "      <td>https://www.coursera.org/learn/silicon-thin-fi...</td>\n",
       "      <td>This course consists of a general presentation...</td>\n",
       "      <td>chemistry  physics  Solar Energy  film  lambda...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance for Managers</td>\n",
       "      <td>IESE Business School</td>\n",
       "      <td>Intermediate</td>\n",
       "      <td>4.8</td>\n",
       "      <td>https://www.coursera.org/learn/operational-fin...</td>\n",
       "      <td>When it comes to numbers, there is always more...</td>\n",
       "      <td>accounts receivable  dupont analysis  analysis...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve Data using Single-Table SQL Queries</td>\n",
       "      <td>Coursera Project Network</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>4.6</td>\n",
       "      <td>https://www.coursera.org/learn/single-table-sq...</td>\n",
       "      <td>In this course you�ll learn how to effectively...</td>\n",
       "      <td>Data Analysis  select (sql)  database manageme...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         Course Name  \\\n",
       "0  Write A Feature Length Screenplay For Film Or ...   \n",
       "1  Business Strategy: Business Model Canvas Analy...   \n",
       "2                      Silicon Thin Film Solar Cells   \n",
       "3                               Finance for Managers   \n",
       "4       Retrieve Data using Single-Table SQL Queries   \n",
       "\n",
       "                  University Difficulty Level Course Rating  \\\n",
       "0  Michigan State University         Beginner           4.8   \n",
       "1   Coursera Project Network         Beginner           4.8   \n",
       "2        �cole Polytechnique         Advanced           4.1   \n",
       "3       IESE Business School     Intermediate           4.8   \n",
       "4   Coursera Project Network         Beginner           4.6   \n",
       "\n",
       "                                          Course URL  \\\n",
       "0  https://www.coursera.org/learn/write-a-feature...   \n",
       "1  https://www.coursera.org/learn/canvas-analysis...   \n",
       "2  https://www.coursera.org/learn/silicon-thin-fi...   \n",
       "3  https://www.coursera.org/learn/operational-fin...   \n",
       "4  https://www.coursera.org/learn/single-table-sq...   \n",
       "\n",
       "                                  Course Description  \\\n",
       "0  Write a Full Length Feature Film Script  In th...   \n",
       "1  By the end of this guided project, you will be...   \n",
       "2  This course consists of a general presentation...   \n",
       "3  When it comes to numbers, there is always more...   \n",
       "4  In this course you�ll learn how to effectively...   \n",
       "\n",
       "                                              Skills  \n",
       "0  Drama  Comedy  peering  screenwriting  film  D...  \n",
       "1  Finance  business plan  persona (user experien...  \n",
       "2  chemistry  physics  Solar Energy  film  lambda...  \n",
       "3  accounts receivable  dupont analysis  analysis...  \n",
       "4  Data Analysis  select (sql)  database manageme...  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"Coursera.csv\")\n",
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "100f65ea",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:44.907504Z",
     "iopub.status.busy": "2022-01-26T14:19:44.906513Z",
     "iopub.status.idle": "2022-01-26T14:19:44.912134Z",
     "shell.execute_reply": "2022-01-26T14:19:44.912600Z",
     "shell.execute_reply.started": "2022-01-26T12:46:50.159391Z"
    },
    "papermill": {
     "duration": 0.059633,
     "end_time": "2022-01-26T14:19:44.912776",
     "exception": false,
     "start_time": "2022-01-26T14:19:44.853143",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3522, 7)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape #3522 courses and 7 columns with different attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0993059a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.041493Z",
     "iopub.status.busy": "2022-01-26T14:19:45.040248Z",
     "iopub.status.idle": "2022-01-26T14:19:45.060303Z",
     "shell.execute_reply": "2022-01-26T14:19:45.061266Z",
     "shell.execute_reply.started": "2022-01-26T12:46:50.515284Z"
    },
    "papermill": {
     "duration": 0.097295,
     "end_time": "2022-01-26T14:19:45.061496",
     "exception": false,
     "start_time": "2022-01-26T14:19:44.964201",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 3522 entries, 0 to 3521\n",
      "Data columns (total 7 columns):\n",
      " #   Column              Non-Null Count  Dtype \n",
      "---  ------              --------------  ----- \n",
      " 0   Course Name         3522 non-null   object\n",
      " 1   University          3522 non-null   object\n",
      " 2   Difficulty Level    3522 non-null   object\n",
      " 3   Course Rating       3522 non-null   object\n",
      " 4   Course URL          3522 non-null   object\n",
      " 5   Course Description  3522 non-null   object\n",
      " 6   Skills              3522 non-null   object\n",
      "dtypes: object(7)\n",
      "memory usage: 192.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "65c2dc88",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.179184Z",
     "iopub.status.busy": "2022-01-26T14:19:45.178159Z",
     "iopub.status.idle": "2022-01-26T14:19:45.189975Z",
     "shell.execute_reply": "2022-01-26T14:19:45.190569Z",
     "shell.execute_reply.started": "2022-01-26T12:46:50.913696Z"
    },
    "papermill": {
     "duration": 0.068015,
     "end_time": "2022-01-26T14:19:45.190768",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.122753",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Course Name           0\n",
       "University            0\n",
       "Difficulty Level      0\n",
       "Course Rating         0\n",
       "Course URL            0\n",
       "Course Description    0\n",
       "Skills                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum() #no value is missing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1fa17aa6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.299135Z",
     "iopub.status.busy": "2022-01-26T14:19:45.298106Z",
     "iopub.status.idle": "2022-01-26T14:19:45.306170Z",
     "shell.execute_reply": "2022-01-26T14:19:45.306688Z",
     "shell.execute_reply.started": "2022-01-26T12:46:53.446023Z"
    },
    "papermill": {
     "duration": 0.064492,
     "end_time": "2022-01-26T14:19:45.306882",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.242390",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Difficulty Level\n",
       "Beginner          1444\n",
       "Advanced          1005\n",
       "Intermediate       837\n",
       "Conversant         186\n",
       "Not Calibrated      50\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Difficulty Level'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "899491c8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.415285Z",
     "iopub.status.busy": "2022-01-26T14:19:45.414312Z",
     "iopub.status.idle": "2022-01-26T14:19:45.422096Z",
     "shell.execute_reply": "2022-01-26T14:19:45.422711Z",
     "shell.execute_reply.started": "2022-01-26T12:46:53.943586Z"
    },
    "papermill": {
     "duration": 0.064131,
     "end_time": "2022-01-26T14:19:45.422886",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.358755",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Course Rating\n",
       "4.7               740\n",
       "4.6               623\n",
       "4.8               598\n",
       "4.5               389\n",
       "4.4               242\n",
       "4.9               180\n",
       "4.3               165\n",
       "4.2               121\n",
       "5                  90\n",
       "4.1                85\n",
       "Not Calibrated     82\n",
       "4                  51\n",
       "3.8                24\n",
       "3.9                20\n",
       "3.6                18\n",
       "3.7                18\n",
       "3.5                17\n",
       "3.4                13\n",
       "3                  12\n",
       "3.2                 9\n",
       "3.3                 6\n",
       "2.9                 6\n",
       "2.6                 2\n",
       "2.8                 2\n",
       "2.4                 2\n",
       "1                   2\n",
       "2                   1\n",
       "3.1                 1\n",
       "2.5                 1\n",
       "1.9                 1\n",
       "2.3                 1\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Course Rating'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "dab334a6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.531483Z",
     "iopub.status.busy": "2022-01-26T14:19:45.530498Z",
     "iopub.status.idle": "2022-01-26T14:19:45.539391Z",
     "shell.execute_reply": "2022-01-26T14:19:45.539891Z",
     "shell.execute_reply.started": "2022-01-26T12:46:54.61919Z"
    },
    "papermill": {
     "duration": 0.064827,
     "end_time": "2022-01-26T14:19:45.540088",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.475261",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "University\n",
       "Coursera Project Network                      562\n",
       "University of Illinois at Urbana-Champaign    138\n",
       "Johns Hopkins University                      110\n",
       "University of Michigan                        101\n",
       "University of Colorado Boulder                101\n",
       "                                             ... \n",
       "GitLab                                          1\n",
       "Yeshiva University                              1\n",
       "University of Glasgow                           1\n",
       "Laureate Education                              1\n",
       "The World Bank Group                            1\n",
       "Name: count, Length: 184, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['University'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cf4e4a14",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.653057Z",
     "iopub.status.busy": "2022-01-26T14:19:45.651968Z",
     "iopub.status.idle": "2022-01-26T14:19:45.657690Z",
     "shell.execute_reply": "2022-01-26T14:19:45.658305Z",
     "shell.execute_reply.started": "2022-01-26T12:46:56.396485Z"
    },
    "papermill": {
     "duration": 0.064954,
     "end_time": "2022-01-26T14:19:45.658504",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.593550",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       Write A Feature Length Screenplay For Film Or ...\n",
       "1       Business Strategy: Business Model Canvas Analy...\n",
       "2                           Silicon Thin Film Solar Cells\n",
       "3                                    Finance for Managers\n",
       "4            Retrieve Data using Single-Table SQL Queries\n",
       "                              ...                        \n",
       "3517    Capstone: Retrieving, Processing, and Visualiz...\n",
       "3518                     Patrick Henry: Forgotten Founder\n",
       "3519    Business intelligence and data analytics: Gene...\n",
       "3520                                  Rigid Body Dynamics\n",
       "3521    Architecting with Google Kubernetes Engine: Pr...\n",
       "Name: Course Name, Length: 3522, dtype: object"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Course Name']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "42f0e96a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:45.985862Z",
     "iopub.status.busy": "2022-01-26T14:19:45.984756Z",
     "iopub.status.idle": "2022-01-26T14:19:45.990667Z",
     "shell.execute_reply": "2022-01-26T14:19:45.991205Z",
     "shell.execute_reply.started": "2022-01-26T12:49:25.396394Z"
    },
    "papermill": {
     "duration": 0.064324,
     "end_time": "2022-01-26T14:19:45.991385",
     "exception": false,
     "start_time": "2022-01-26T14:19:45.927061",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = data[['Course Name','Difficulty Level','Course Description','Skills']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "04cc92ed",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:46.104150Z",
     "iopub.status.busy": "2022-01-26T14:19:46.103123Z",
     "iopub.status.idle": "2022-01-26T14:19:46.113861Z",
     "shell.execute_reply": "2022-01-26T14:19:46.114401Z",
     "shell.execute_reply.started": "2022-01-26T12:49:30.127057Z"
    },
    "papermill": {
     "duration": 0.067641,
     "end_time": "2022-01-26T14:19:46.114589",
     "exception": false,
     "start_time": "2022-01-26T14:19:46.046948",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Course Name</th>\n",
       "      <th>Difficulty Level</th>\n",
       "      <th>Course Description</th>\n",
       "      <th>Skills</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write A Feature Length Screenplay For Film Or ...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>Write a Full Length Feature Film Script  In th...</td>\n",
       "      <td>Drama  Comedy  peering  screenwriting  film  D...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business Strategy: Business Model Canvas Analy...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>By the end of this guided project, you will be...</td>\n",
       "      <td>Finance  business plan  persona (user experien...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon Thin Film Solar Cells</td>\n",
       "      <td>Advanced</td>\n",
       "      <td>This course consists of a general presentation...</td>\n",
       "      <td>chemistry  physics  Solar Energy  film  lambda...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance for Managers</td>\n",
       "      <td>Intermediate</td>\n",
       "      <td>When it comes to numbers, there is always more...</td>\n",
       "      <td>accounts receivable  dupont analysis  analysis...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve Data using Single-Table SQL Queries</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>In this course you�ll learn how to effectively...</td>\n",
       "      <td>Data Analysis  select (sql)  database manageme...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         Course Name Difficulty Level  \\\n",
       "0  Write A Feature Length Screenplay For Film Or ...         Beginner   \n",
       "1  Business Strategy: Business Model Canvas Analy...         Beginner   \n",
       "2                      Silicon Thin Film Solar Cells         Advanced   \n",
       "3                               Finance for Managers     Intermediate   \n",
       "4       Retrieve Data using Single-Table SQL Queries         Beginner   \n",
       "\n",
       "                                  Course Description  \\\n",
       "0  Write a Full Length Feature Film Script  In th...   \n",
       "1  By the end of this guided project, you will be...   \n",
       "2  This course consists of a general presentation...   \n",
       "3  When it comes to numbers, there is always more...   \n",
       "4  In this course you�ll learn how to effectively...   \n",
       "\n",
       "                                              Skills  \n",
       "0  Drama  Comedy  peering  screenwriting  film  D...  \n",
       "1  Finance  business plan  persona (user experien...  \n",
       "2  chemistry  physics  Solar Energy  film  lambda...  \n",
       "3  accounts receivable  dupont analysis  analysis...  \n",
       "4  Data Analysis  select (sql)  database manageme...  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e0df7188",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:46.339162Z",
     "iopub.status.busy": "2022-01-26T14:19:46.338482Z",
     "iopub.status.idle": "2022-01-26T14:19:46.453299Z",
     "shell.execute_reply": "2022-01-26T14:19:46.453797Z",
     "shell.execute_reply.started": "2022-01-26T12:53:40.943775Z"
    },
    "papermill": {
     "duration": 0.177814,
     "end_time": "2022-01-26T14:19:46.453980",
     "exception": false,
     "start_time": "2022-01-26T14:19:46.276166",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  # Remove the CWD from sys.path while we load stuff.\n",
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:11: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  # This is added back by InteractiveShellApp.init_path()\n",
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:14: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  \n",
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:15: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  from ipykernel import kernelapp as app\n"
     ]
    }
   ],
   "source": [
    "# Removing spaces between the words (Lambda funtions can be used as well)\n",
    "\n",
    "data['Course Name'] = data['Course Name'].str.replace(' ',',')\n",
    "data['Course Name'] = data['Course Name'].str.replace(',,',',')\n",
    "data['Course Name'] = data['Course Name'].str.replace(':','')\n",
    "data['Course Description'] = data['Course Description'].str.replace(' ',',')\n",
    "data['Course Description'] = data['Course Description'].str.replace(',,',',')\n",
    "data['Course Description'] = data['Course Description'].str.replace('_','')\n",
    "data['Course Description'] = data['Course Description'].str.replace(':','')\n",
    "data['Course Description'] = data['Course Description'].str.replace('(','')\n",
    "data['Course Description'] = data['Course Description'].str.replace(')','')\n",
    "\n",
    "#removing paranthesis from skills columns \n",
    "data['Skills'] = data['Skills'].str.replace('(','')\n",
    "data['Skills'] = data['Skills'].str.replace(')','')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "624470d3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:46.566735Z",
     "iopub.status.busy": "2022-01-26T14:19:46.565969Z",
     "iopub.status.idle": "2022-01-26T14:19:46.576143Z",
     "shell.execute_reply": "2022-01-26T14:19:46.576719Z",
     "shell.execute_reply.started": "2022-01-26T12:53:53.653156Z"
    },
    "papermill": {
     "duration": 0.068556,
     "end_time": "2022-01-26T14:19:46.576919",
     "exception": false,
     "start_time": "2022-01-26T14:19:46.508363",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Course Name</th>\n",
       "      <th>Difficulty Level</th>\n",
       "      <th>Course Description</th>\n",
       "      <th>Skills</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write,A,Feature,Length,Screenplay,For,Film,Or,...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>Write,a,Full,Length,Feature,Film,Script,In,thi...</td>\n",
       "      <td>Drama  Comedy  peering  screenwriting  film  D...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business,Strategy,Business,Model,Canvas,Analys...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>By,the,end,of,this,guided,project,you,will,be,...</td>\n",
       "      <td>Finance  business plan  persona user experienc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon,Thin,Film,Solar,Cells</td>\n",
       "      <td>Advanced</td>\n",
       "      <td>This,course,consists,of,a,general,presentation...</td>\n",
       "      <td>chemistry  physics  Solar Energy  film  lambda...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance,for,Managers</td>\n",
       "      <td>Intermediate</td>\n",
       "      <td>When,it,comes,to,numbers,there,is,always,more,...</td>\n",
       "      <td>accounts receivable  dupont analysis  analysis...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve,Data,using,Single-Table,SQL,Queries</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>In,this,course,you�ll,learn,how,to,effectively...</td>\n",
       "      <td>Data Analysis  select sql  database management...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         Course Name Difficulty Level  \\\n",
       "0  Write,A,Feature,Length,Screenplay,For,Film,Or,...         Beginner   \n",
       "1  Business,Strategy,Business,Model,Canvas,Analys...         Beginner   \n",
       "2                      Silicon,Thin,Film,Solar,Cells         Advanced   \n",
       "3                               Finance,for,Managers     Intermediate   \n",
       "4       Retrieve,Data,using,Single-Table,SQL,Queries         Beginner   \n",
       "\n",
       "                                  Course Description  \\\n",
       "0  Write,a,Full,Length,Feature,Film,Script,In,thi...   \n",
       "1  By,the,end,of,this,guided,project,you,will,be,...   \n",
       "2  This,course,consists,of,a,general,presentation...   \n",
       "3  When,it,comes,to,numbers,there,is,always,more,...   \n",
       "4  In,this,course,you�ll,learn,how,to,effectively...   \n",
       "\n",
       "                                              Skills  \n",
       "0  Drama  Comedy  peering  screenwriting  film  D...  \n",
       "1  Finance  business plan  persona user experienc...  \n",
       "2  chemistry  physics  Solar Energy  film  lambda...  \n",
       "3  accounts receivable  dupont analysis  analysis...  \n",
       "4  Data Analysis  select sql  database management...  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7efa10a9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:46.805542Z",
     "iopub.status.busy": "2022-01-26T14:19:46.804854Z",
     "iopub.status.idle": "2022-01-26T14:19:46.823188Z",
     "shell.execute_reply": "2022-01-26T14:19:46.822651Z",
     "shell.execute_reply.started": "2022-01-26T12:55:58.319873Z"
    },
    "papermill": {
     "duration": 0.076499,
     "end_time": "2022-01-26T14:19:46.823340",
     "exception": false,
     "start_time": "2022-01-26T14:19:46.746841",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "eee80759",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:46.937064Z",
     "iopub.status.busy": "2022-01-26T14:19:46.936365Z",
     "iopub.status.idle": "2022-01-26T14:19:46.948262Z",
     "shell.execute_reply": "2022-01-26T14:19:46.948878Z",
     "shell.execute_reply.started": "2022-01-26T12:56:03.509244Z"
    },
    "papermill": {
     "duration": 0.070316,
     "end_time": "2022-01-26T14:19:46.949049",
     "exception": false,
     "start_time": "2022-01-26T14:19:46.878733",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Course Name</th>\n",
       "      <th>Difficulty Level</th>\n",
       "      <th>Course Description</th>\n",
       "      <th>Skills</th>\n",
       "      <th>tags</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write,A,Feature,Length,Screenplay,For,Film,Or,...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>Write,a,Full,Length,Feature,Film,Script,In,thi...</td>\n",
       "      <td>Drama  Comedy  peering  screenwriting  film  D...</td>\n",
       "      <td>Write,A,Feature,Length,Screenplay,For,Film,Or,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business,Strategy,Business,Model,Canvas,Analys...</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>By,the,end,of,this,guided,project,you,will,be,...</td>\n",
       "      <td>Finance  business plan  persona user experienc...</td>\n",
       "      <td>Business,Strategy,Business,Model,Canvas,Analys...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon,Thin,Film,Solar,Cells</td>\n",
       "      <td>Advanced</td>\n",
       "      <td>This,course,consists,of,a,general,presentation...</td>\n",
       "      <td>chemistry  physics  Solar Energy  film  lambda...</td>\n",
       "      <td>Silicon,Thin,Film,Solar,CellsAdvancedThis,cour...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance,for,Managers</td>\n",
       "      <td>Intermediate</td>\n",
       "      <td>When,it,comes,to,numbers,there,is,always,more,...</td>\n",
       "      <td>accounts receivable  dupont analysis  analysis...</td>\n",
       "      <td>Finance,for,ManagersIntermediateWhen,it,comes,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve,Data,using,Single-Table,SQL,Queries</td>\n",
       "      <td>Beginner</td>\n",
       "      <td>In,this,course,you�ll,learn,how,to,effectively...</td>\n",
       "      <td>Data Analysis  select sql  database management...</td>\n",
       "      <td>Retrieve,Data,using,Single-Table,SQL,QueriesBe...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         Course Name Difficulty Level  \\\n",
       "0  Write,A,Feature,Length,Screenplay,For,Film,Or,...         Beginner   \n",
       "1  Business,Strategy,Business,Model,Canvas,Analys...         Beginner   \n",
       "2                      Silicon,Thin,Film,Solar,Cells         Advanced   \n",
       "3                               Finance,for,Managers     Intermediate   \n",
       "4       Retrieve,Data,using,Single-Table,SQL,Queries         Beginner   \n",
       "\n",
       "                                  Course Description  \\\n",
       "0  Write,a,Full,Length,Feature,Film,Script,In,thi...   \n",
       "1  By,the,end,of,this,guided,project,you,will,be,...   \n",
       "2  This,course,consists,of,a,general,presentation...   \n",
       "3  When,it,comes,to,numbers,there,is,always,more,...   \n",
       "4  In,this,course,you�ll,learn,how,to,effectively...   \n",
       "\n",
       "                                              Skills  \\\n",
       "0  Drama  Comedy  peering  screenwriting  film  D...   \n",
       "1  Finance  business plan  persona user experienc...   \n",
       "2  chemistry  physics  Solar Energy  film  lambda...   \n",
       "3  accounts receivable  dupont analysis  analysis...   \n",
       "4  Data Analysis  select sql  database management...   \n",
       "\n",
       "                                                tags  \n",
       "0  Write,A,Feature,Length,Screenplay,For,Film,Or,...  \n",
       "1  Business,Strategy,Business,Model,Canvas,Analys...  \n",
       "2  Silicon,Thin,Film,Solar,CellsAdvancedThis,cour...  \n",
       "3  Finance,for,ManagersIntermediateWhen,it,comes,...  \n",
       "4  Retrieve,Data,using,Single-Table,SQL,QueriesBe...  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "66879f0c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.064568Z",
     "iopub.status.busy": "2022-01-26T14:19:47.063864Z",
     "iopub.status.idle": "2022-01-26T14:19:47.068955Z",
     "shell.execute_reply": "2022-01-26T14:19:47.069577Z",
     "shell.execute_reply.started": "2022-01-26T12:56:19.913202Z"
    },
    "papermill": {
     "duration": 0.064563,
     "end_time": "2022-01-26T14:19:47.069883",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.005320",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Business,Strategy,Business,Model,Canvas,Analysis,with,MiroBeginnerBy,the,end,of,this,guided,project,you,will,be,fluent,in,identifying,and,creating,Business,Model,Canvas,solutions,based,on,previous,high-level,analyses,and,research,data.,This,will,enable,you,to,identify,and,map,the,elements,required,for,new,products,and,services.,Furthermore,it,is,essential,for,generating,positive,results,for,your,business,venture.,This,guided,project,is,designed,to,engage,and,harness,your,visionary,and,exploratory,abilities.,You,will,use,proven,models,in,strategy,and,product,development,with,the,Miro,platform,to,explore,and,analyse,your,business,propositions.,,We,will,practice,critically,examining,results,from,previous,analysis,and,research,results,in,deriving,the,values,for,each,of,the,business,model,sections.Finance  business plan  persona user experience  business model canvas  Planning  Business  project  Product Development  presentation  Strategy business business-strategy'"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['tags'].iloc[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b3807261",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.295634Z",
     "iopub.status.busy": "2022-01-26T14:19:47.294976Z",
     "iopub.status.idle": "2022-01-26T14:19:47.300427Z",
     "shell.execute_reply": "2022-01-26T14:19:47.300988Z",
     "shell.execute_reply.started": "2022-01-26T12:56:47.650782Z"
    },
    "papermill": {
     "duration": 0.064683,
     "end_time": "2022-01-26T14:19:47.301174",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.236491",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "new_df = data[['Course Name','tags']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e4f516b3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.423897Z",
     "iopub.status.busy": "2022-01-26T14:19:47.423181Z",
     "iopub.status.idle": "2022-01-26T14:19:47.425800Z",
     "shell.execute_reply": "2022-01-26T14:19:47.426278Z",
     "shell.execute_reply.started": "2022-01-26T12:56:53.601235Z"
    },
    "papermill": {
     "duration": 0.069032,
     "end_time": "2022-01-26T14:19:47.426446",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.357414",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Course Name</th>\n",
       "      <th>tags</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write,A,Feature,Length,Screenplay,For,Film,Or,...</td>\n",
       "      <td>Write,A,Feature,Length,Screenplay,For,Film,Or,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business,Strategy,Business,Model,Canvas,Analys...</td>\n",
       "      <td>Business,Strategy,Business,Model,Canvas,Analys...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon,Thin,Film,Solar,Cells</td>\n",
       "      <td>Silicon,Thin,Film,Solar,CellsAdvancedThis,cour...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance,for,Managers</td>\n",
       "      <td>Finance,for,ManagersIntermediateWhen,it,comes,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve,Data,using,Single-Table,SQL,Queries</td>\n",
       "      <td>Retrieve,Data,using,Single-Table,SQL,QueriesBe...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         Course Name  \\\n",
       "0  Write,A,Feature,Length,Screenplay,For,Film,Or,...   \n",
       "1  Business,Strategy,Business,Model,Canvas,Analys...   \n",
       "2                      Silicon,Thin,Film,Solar,Cells   \n",
       "3                               Finance,for,Managers   \n",
       "4       Retrieve,Data,using,Single-Table,SQL,Queries   \n",
       "\n",
       "                                                tags  \n",
       "0  Write,A,Feature,Length,Screenplay,For,Film,Or,...  \n",
       "1  Business,Strategy,Business,Model,Canvas,Analys...  \n",
       "2  Silicon,Thin,Film,Solar,CellsAdvancedThis,cour...  \n",
       "3  Finance,for,ManagersIntermediateWhen,it,comes,...  \n",
       "4  Retrieve,Data,using,Single-Table,SQL,QueriesBe...  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f716f14c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.543759Z",
     "iopub.status.busy": "2022-01-26T14:19:47.543010Z",
     "iopub.status.idle": "2022-01-26T14:19:47.568840Z",
     "shell.execute_reply": "2022-01-26T14:19:47.569353Z",
     "shell.execute_reply.started": "2022-01-26T12:57:29.758901Z"
    },
    "papermill": {
     "duration": 0.08545,
     "end_time": "2022-01-26T14:19:47.569554",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.484104",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "new_df['tags'] = data['tags'].str.replace(',',' ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "38d9e830",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.688611Z",
     "iopub.status.busy": "2022-01-26T14:19:47.687894Z",
     "iopub.status.idle": "2022-01-26T14:19:47.695125Z",
     "shell.execute_reply": "2022-01-26T14:19:47.695760Z",
     "shell.execute_reply.started": "2022-01-26T12:58:08.296915Z"
    },
    "papermill": {
     "duration": 0.069347,
     "end_time": "2022-01-26T14:19:47.695936",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.626589",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "new_df['Course Name'] = data['Course Name'].str.replace(',',' ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3af16b3c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.817556Z",
     "iopub.status.busy": "2022-01-26T14:19:47.816533Z",
     "iopub.status.idle": "2022-01-26T14:19:47.827197Z",
     "shell.execute_reply": "2022-01-26T14:19:47.827814Z",
     "shell.execute_reply.started": "2022-01-26T12:58:11.491014Z"
    },
    "papermill": {
     "duration": 0.073413,
     "end_time": "2022-01-26T14:19:47.828002",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.754589",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py:5047: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  errors=errors,\n"
     ]
    }
   ],
   "source": [
    "new_df.rename(columns = {'Course Name':'course_name'}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9c65075a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:47.949931Z",
     "iopub.status.busy": "2022-01-26T14:19:47.948954Z",
     "iopub.status.idle": "2022-01-26T14:19:47.977325Z",
     "shell.execute_reply": "2022-01-26T14:19:47.977969Z",
     "shell.execute_reply.started": "2022-01-26T12:58:52.731609Z"
    },
    "papermill": {
     "duration": 0.090638,
     "end_time": "2022-01-26T14:19:47.978152",
     "exception": false,
     "start_time": "2022-01-26T14:19:47.887514",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #lower casing the tags column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b2bc5584",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:48.112723Z",
     "iopub.status.busy": "2022-01-26T14:19:48.112045Z",
     "iopub.status.idle": "2022-01-26T14:19:48.121076Z",
     "shell.execute_reply": "2022-01-26T14:19:48.121592Z",
     "shell.execute_reply.started": "2022-01-26T12:58:55.635891Z"
    },
    "papermill": {
     "duration": 0.069732,
     "end_time": "2022-01-26T14:19:48.121783",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.052051",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>course_name</th>\n",
       "      <th>tags</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Write A Feature Length Screenplay For Film Or ...</td>\n",
       "      <td>write a feature length screenplay for film or ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Business Strategy Business Model Canvas Analys...</td>\n",
       "      <td>business strategy business model canvas analys...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Silicon Thin Film Solar Cells</td>\n",
       "      <td>silicon thin film solar cellsadvancedthis cour...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Finance for Managers</td>\n",
       "      <td>finance for managersintermediatewhen it comes ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Retrieve Data using Single-Table SQL Queries</td>\n",
       "      <td>retrieve data using single-table sql queriesbe...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         course_name  \\\n",
       "0  Write A Feature Length Screenplay For Film Or ...   \n",
       "1  Business Strategy Business Model Canvas Analys...   \n",
       "2                      Silicon Thin Film Solar Cells   \n",
       "3                               Finance for Managers   \n",
       "4       Retrieve Data using Single-Table SQL Queries   \n",
       "\n",
       "                                                tags  \n",
       "0  write a feature length screenplay for film or ...  \n",
       "1  business strategy business model canvas analys...  \n",
       "2  silicon thin film solar cellsadvancedthis cour...  \n",
       "3  finance for managersintermediatewhen it comes ...  \n",
       "4  retrieve data using single-table sql queriesbe...  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "66db7225",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:48.240478Z",
     "iopub.status.busy": "2022-01-26T14:19:48.239827Z",
     "iopub.status.idle": "2022-01-26T14:19:48.244379Z",
     "shell.execute_reply": "2022-01-26T14:19:48.244875Z",
     "shell.execute_reply.started": "2022-01-26T12:59:36.995288Z"
    },
    "papermill": {
     "duration": 0.065828,
     "end_time": "2022-01-26T14:19:48.245060",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.179232",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3522, 2)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.shape #3522 courses with tags and 2 columns (course_name and tags)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4a5083c5",
   "metadata": {
    "papermill": {
     "duration": 0.060035,
     "end_time": "2022-01-26T14:19:48.362969",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.302934",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "4e13af12",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:48.484188Z",
     "iopub.status.busy": "2022-01-26T14:19:48.483520Z",
     "iopub.status.idle": "2022-01-26T14:19:48.622802Z",
     "shell.execute_reply": "2022-01-26T14:19:48.622207Z",
     "shell.execute_reply.started": "2022-01-26T13:00:20.637145Z"
    },
    "papermill": {
     "duration": 0.201577,
     "end_time": "2022-01-26T14:19:48.622967",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.421390",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Text Vectorization\n",
    "from sklearn.feature_extraction.text import CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "e6372627",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:48.746790Z",
     "iopub.status.busy": "2022-01-26T14:19:48.746084Z",
     "iopub.status.idle": "2022-01-26T14:19:48.748607Z",
     "shell.execute_reply": "2022-01-26T14:19:48.748074Z",
     "shell.execute_reply.started": "2022-01-26T13:00:25.765271Z"
    },
    "papermill": {
     "duration": 0.06615,
     "end_time": "2022-01-26T14:19:48.748745",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.682595",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cv = CountVectorizer(max_features=5000,stop_words='english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7d43550e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:48.868706Z",
     "iopub.status.busy": "2022-01-26T14:19:48.868015Z",
     "iopub.status.idle": "2022-01-26T14:19:49.836960Z",
     "shell.execute_reply": "2022-01-26T14:19:49.837458Z",
     "shell.execute_reply.started": "2022-01-26T13:00:30.470945Z"
    },
    "papermill": {
     "duration": 1.030812,
     "end_time": "2022-01-26T14:19:49.837694",
     "exception": false,
     "start_time": "2022-01-26T14:19:48.806882",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "vectors = cv.fit_transform(new_df['tags']).toarray()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cfa085e",
   "metadata": {
    "papermill": {
     "duration": 0.059295,
     "end_time": "2022-01-26T14:19:49.956538",
     "exception": false,
     "start_time": "2022-01-26T14:19:49.897243",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Stemming Process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "766e85f8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:50.079154Z",
     "iopub.status.busy": "2022-01-26T14:19:50.078376Z",
     "iopub.status.idle": "2022-01-26T14:19:50.720939Z",
     "shell.execute_reply": "2022-01-26T14:19:50.720340Z",
     "shell.execute_reply.started": "2022-01-26T13:00:50.236008Z"
    },
    "papermill": {
     "duration": 0.70524,
     "end_time": "2022-01-26T14:19:50.721119",
     "exception": false,
     "start_time": "2022-01-26T14:19:50.015879",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import nltk #for stemming process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "50725a4b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:50.844107Z",
     "iopub.status.busy": "2022-01-26T14:19:50.843400Z",
     "iopub.status.idle": "2022-01-26T14:19:50.847403Z",
     "shell.execute_reply": "2022-01-26T14:19:50.847932Z",
     "shell.execute_reply.started": "2022-01-26T13:00:55.372687Z"
    },
    "papermill": {
     "duration": 0.066187,
     "end_time": "2022-01-26T14:19:50.848117",
     "exception": false,
     "start_time": "2022-01-26T14:19:50.781930",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from nltk.stem.porter import PorterStemmer\n",
    "ps = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "9710abff",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:50.970441Z",
     "iopub.status.busy": "2022-01-26T14:19:50.969372Z",
     "iopub.status.idle": "2022-01-26T14:19:50.974055Z",
     "shell.execute_reply": "2022-01-26T14:19:50.974518Z",
     "shell.execute_reply.started": "2022-01-26T13:01:26.727138Z"
    },
    "papermill": {
     "duration": 0.067833,
     "end_time": "2022-01-26T14:19:50.974701",
     "exception": false,
     "start_time": "2022-01-26T14:19:50.906868",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#defining the stemming function\n",
    "def stem(text):\n",
    "    y=[]\n",
    "    \n",
    "    for i in text.split():\n",
    "        y.append(ps.stem(i))\n",
    "    \n",
    "    return \" \".join(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "51c560be",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:19:51.096458Z",
     "iopub.status.busy": "2022-01-26T14:19:51.095435Z",
     "iopub.status.idle": "2022-01-26T14:20:09.833321Z",
     "shell.execute_reply": "2022-01-26T14:20:09.834233Z",
     "shell.execute_reply.started": "2022-01-26T13:02:02.700396Z"
    },
    "papermill": {
     "duration": 18.800875,
     "end_time": "2022-01-26T14:20:09.834414",
     "exception": false,
     "start_time": "2022-01-26T14:19:51.033539",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": [
    "new_df['tags'] = new_df['tags'].apply(stem) #applying stemming on the tags column"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6ee86c9",
   "metadata": {
    "papermill": {
     "duration": 0.059879,
     "end_time": "2022-01-26T14:20:09.952432",
     "exception": false,
     "start_time": "2022-01-26T14:20:09.892553",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Similarity Measure "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9b4f80e4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:10.075324Z",
     "iopub.status.busy": "2022-01-26T14:20:10.074250Z",
     "iopub.status.idle": "2022-01-26T14:20:10.078349Z",
     "shell.execute_reply": "2022-01-26T14:20:10.078920Z",
     "shell.execute_reply.started": "2022-01-26T13:02:27.946429Z"
    },
    "papermill": {
     "duration": 0.067769,
     "end_time": "2022-01-26T14:20:10.079103",
     "exception": false,
     "start_time": "2022-01-26T14:20:10.011334",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "26fcc651",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:10.201244Z",
     "iopub.status.busy": "2022-01-26T14:20:10.200570Z",
     "iopub.status.idle": "2022-01-26T14:20:11.472233Z",
     "shell.execute_reply": "2022-01-26T14:20:11.473186Z",
     "shell.execute_reply.started": "2022-01-26T13:02:34.205466Z"
    },
    "papermill": {
     "duration": 1.335405,
     "end_time": "2022-01-26T14:20:11.473497",
     "exception": false,
     "start_time": "2022-01-26T14:20:10.138092",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "similarity = cosine_similarity(vectors)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "100bd489",
   "metadata": {
    "papermill": {
     "duration": 0.059252,
     "end_time": "2022-01-26T14:20:11.619088",
     "exception": false,
     "start_time": "2022-01-26T14:20:11.559836",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Recommendation Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b0e15b2b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:11.745691Z",
     "iopub.status.busy": "2022-01-26T14:20:11.744935Z",
     "iopub.status.idle": "2022-01-26T14:20:11.746394Z",
     "shell.execute_reply": "2022-01-26T14:20:11.746994Z",
     "shell.execute_reply.started": "2022-01-26T13:03:26.756125Z"
    },
    "papermill": {
     "duration": 0.068387,
     "end_time": "2022-01-26T14:20:11.747173",
     "exception": false,
     "start_time": "2022-01-26T14:20:11.678786",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def recommend(course):\n",
    "    course_index = new_df[new_df['course_name'] == course].index[0]\n",
    "    distances = similarity[course_index]\n",
    "    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]\n",
    "    \n",
    "    for i in course_list:\n",
    "        print(new_df.iloc[i[0]].course_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "0ba0466e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:11.872210Z",
     "iopub.status.busy": "2022-01-26T14:20:11.871269Z",
     "iopub.status.idle": "2022-01-26T14:20:11.884642Z",
     "shell.execute_reply": "2022-01-26T14:20:11.885341Z",
     "shell.execute_reply.started": "2022-01-26T13:03:28.042547Z"
    },
    "papermill": {
     "duration": 0.076619,
     "end_time": "2022-01-26T14:20:11.885608",
     "exception": false,
     "start_time": "2022-01-26T14:20:11.808989",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Product Development Customer Persona Development with Miro\n",
      "Product and Service Development Empathy Mapping with Miro\n",
      "Product Development Customer Journey Mapping with Miro\n",
      "Analyzing Macro-Environmental Factors Using Creately\n",
      "Business Strategy in Practice (Project-centered Course)\n",
      "Innovating with the Business Model Canvas\n"
     ]
    }
   ],
   "source": [
    "recommend('Business Strategy Business Model Canvas Analysis with Miro') "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf9e8456",
   "metadata": {
    "papermill": {
     "duration": 0.060079,
     "end_time": "2022-01-26T14:20:12.007234",
     "exception": false,
     "start_time": "2022-01-26T14:20:11.947155",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**So these are the 6 courses which are recommended based on our search in the recommendation function**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "973c1cdf",
   "metadata": {
    "papermill": {
     "duration": 0.060412,
     "end_time": "2022-01-26T14:20:12.130219",
     "exception": false,
     "start_time": "2022-01-26T14:20:12.069807",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exporting the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "949dc95b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:12.253179Z",
     "iopub.status.busy": "2022-01-26T14:20:12.252427Z",
     "iopub.status.idle": "2022-01-26T14:20:12.255671Z",
     "shell.execute_reply": "2022-01-26T14:20:12.256173Z",
     "shell.execute_reply.started": "2022-01-26T13:04:30.184632Z"
    },
    "papermill": {
     "duration": 0.066666,
     "end_time": "2022-01-26T14:20:12.256353",
     "exception": false,
     "start_time": "2022-01-26T14:20:12.189687",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6c7ef9d3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-26T14:20:12.386331Z",
     "iopub.status.busy": "2022-01-26T14:20:12.385507Z",
     "iopub.status.idle": "2022-01-26T14:20:12.388317Z",
     "shell.execute_reply": "2022-01-26T14:20:12.388854Z",
     "shell.execute_reply.started": "2022-01-26T13:05:01.974915Z"
    },
    "papermill": {
     "duration": 0.068209,
     "end_time": "2022-01-26T14:20:12.389043",
     "exception": false,
     "start_time": "2022-01-26T14:20:12.320834",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# pickle.dump(similarity,open('similarity.pkl','wb'))\n",
    "# pickle.dump(new_df.to_dict(),open('course_list.pkl','wb')) #contains the dataframe in dict \n",
    "# pickle.dump(new_df,open('courses.pkl','wb'))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 40.65955,
   "end_time": "2022-01-26T14:20:13.730648",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-01-26T14:19:33.071098",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
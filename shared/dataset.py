import scipy.io as sio
import numpy as np
import datetime

from random import shuffle

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DatasetDownloader(metaclass=Singleton):
    def __init__(self):
        self.file_list = list()

    def add_file(self, dir, url, extract_to):
        self.file_list.append({
            'dir': dir,
            'url': dir,
            'extract_to': dir,
        })

    def download(self):
        for f in self.file_list:
            print(f)

class SoF():
    def load_annotations(self, path, setname):
        self.crop_enum = ('nc', 'cr')
        self.emotion_enum = ('no', 'hp', 'sd', 'sr')
        self.occ_enum = ('e0', 'en', 'em')
        self.filter_enum = ('nl', 'Gn', 'Gs', 'Ps')
        self.difficulty_enum = ('o', 'e', 'm', 'h')
        self.annotations = sio.loadmat(path)['metadata'][0]
        np.random.shuffle(self.annotations)
        print('Annotations loaded', len(self.annotations))

    def shuffle(self):
        np.random.shuffle(self.annotations)

    def get_face(self, index):
        #'(\w{4})_(\d{5})_([mfMF])_(\d{2})(_([ioIO])_(fr|nf)_(cr|nc)_(no|hp|sd|sr)_(\d{4})_(\d)_(e0|en|em)_(nl|Gn|Gs|Ps)_([oemh]))*\.jpg'
        if index < 0 or index >=  len(self.annotations):
            print('Face index out of range')
            return None
        anno = self.annotations[index]
        sid = anno[0][0][0][0]
        seq = anno[1][0][0][0]
        gender = anno[2][0][0]
        age = anno[3][0][0]
        lighting = anno[4][0][0]
        view = anno[5][0]
        cropped = self.crop_enum[anno[6][0][0]]
        emotion = self.emotion_enum[anno[7][0][0]-1]
        year = anno[8][0][0]
        part = anno[9][0]
        glasses = anno[10][0][0]
        scarf = anno[11][0][0]
        points = anno[12][0]
        est_points = anno[13][0]
        rect = anno[14][0]
        g_rect = anno[15][0]
        illum_quality = anno[16][0][0]
        filename = anno[16][0][0]
        #difficulty = self.difficulty_enum[anno[12][0]]
        #difficulty = anno[12][0]

        file_prefix = '_'.join((sid, seq, gender, str(age), lighting, view, cropped, emotion, str(year), part))
        #print(anno)
        return {
            'sid': sid,
            'seq': seq,
            'gender': gender,
            'age': age,
            'lighting': lighting,
            'view': view, # 5
            'cropped': cropped,
            'emotion': emotion,
            'year': year,
            'part': part,
            'glasses': glasses, # 10
            'scarf': scarf,
            'points': points,
            'est_points': est_points,
            'rect': rect,
            'g_rect': g_rect, #15
            'illum_quality': illum_quality,
            'file_prefix': file_prefix, 
        }

class WikiImdb():
    @staticmethod
    def from_ml_date(matlab_datenum):
        matlab_datenum = int(matlab_datenum)
        date = datetime.date.fromordinal(matlab_datenum) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
        return date.toordinal()

    @staticmethod
    def age(dob, pot):
        pot = datetime.date(pot, 7, 1)
        return pot.toordinal() - dob

    def load_annotations(self, path, setname):
        # The format is for  WIKI faces dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
        annotations = sio.loadmat(path)[setname][0]
        if annotations is None:
            raise IOError('Error loading annotations')
        self.dob = annotations['dob'][0][0]
        self.photo_taken = annotations['photo_taken'][0][0]
        self.full_path = annotations['full_path'][0][0]
        self.gender = annotations['gender'][0][0]
        self.name = annotations['name'][0][0]
        self.face_location = annotations['face_location'][0][0]
        self.face_score = annotations['face_score'][0][0]
        self.second_face_score = annotations['second_face_score'][0][0]
        print('Annotations loaded', len(self.dob))

    def get_face(self, id):
        return (
            id,
            self.dob[id], # date of birth (Matlab serial date number)
            self.photo_taken[id], # year when the photo was taken
            self.full_path[id], # path to file
            self.gender[id], # 0 for female and 1 for male, NaN if unknown
            self.name[id], # name of the celebrity
            self.face_location[id], # location of the face. To crop the face in Matlab run
                               # img(face_location(2):face_location(4),face_location(1):face_location(3),:))
            self.face_score[id], # detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
            self.second_face_score[id], # detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
            #cname = annotations['wiki'][0][0][8][id], # list of all celebrity names. IMDB only
            #cid = annotations['wiki'][0][0][9][id], # index of celebrity name. IMDB only
        )


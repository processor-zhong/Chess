"""Human-readable definitions for the deterministic Plan A feature schema."""
import csv
from pathlib import Path

def write_feature_dictionary(root, c):
    root = Path(root)
    base={
    'rating_diff':('白方等级分减黑方等级分','rating points'),
    'avg_rating':('双方等级分平均值','rating points'),
    'abs_rating_diff':('双方等级分差绝对值','rating points'),
    'base_time_min':('基础时间','minutes'),
    'increment_seconds':('每步增加的秒数','seconds'),
    'rated':('是否为计等级分对局，0/1','binary'),
    'material_diff':('白方子力和减黑方子力和；兵1马象3车5后9','pawn units'),
    'side_to_move_in_check':('第20个半回合后轮到白方，白方是否被将军','binary')}
    suffix={
    'pawn_count':('兵数量','count'),'knight_count':('马数量','count'),'bishop_count':('象数量','count'),
    'rook_count':('车数量','count'),'queen_count':('后数量','count'),
    'piece_count':('含王的全部棋子数量','count'),
    'material':('子力和：兵1马象3车5后9，不计王','pawn units'),
    'minor_off_home_rank':('当前位于己方底线以外的马象数；发展代理量，不追踪棋子个体','count'),
    'center_occupancy':('占据中心 d4/e4/d5/e5 的棋子数','count'),
    'center_attacked_squares':('攻击或防守到的中心格数，含被己方占据的格','count'),
    'attacked_squares':('攻击或防守到的不同棋盘格数；不等于合法着法数','count'),
    'pawn_advancement':('各兵相对初始兵线前进格数的总和，已被吃的兵不计','squares'),
    'castled':('前20个半回合内是否实际易位','binary'),
    'kingside_right':('截断局面是否仍具有王翼易位权','binary'),
    'queenside_right':('截断局面是否仍具有后翼易位权','binary'),
    'king_file':('王所在列，a=0至h=7','index'),
    'king_rank':('王所在行，第1行=0至第8行=7','index'),
    'captures_first20':('前20个半回合该方吃子次数，含吃过路兵','count'),
    'checks_given_first20':('前20个半回合该方走棋后将军次数','count')}
    rows=[]
    for f in c['combined_features']:
     if f in base:desc,unit=base[f]
     else:
      side,s=f.split('_',1);desc,unit=suffix[s];desc=('白方：' if side=='white' else '黑方：')+desc
     rows.append({'feature':f,'group':'pregame' if f in c['pregame_features'] else 'opening_ply20',
                  'definition_zh':desc,'unit':unit,'constant_on_train':f in c['constant_features_on_train'],
                  'uses_future_moves':False})
    with (root/'tables/plan_a20/feature_dictionary.csv').open('w',newline='') as h:
     w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

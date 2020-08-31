# calc_bags_num():商品とエコバックの容量から袋の枚数を計算する関数
# input item_capa:商品の容量(L:liter)
# input eco_bag_capa, s_bag_capa, l_bag_capa:エコバッグ,S,Lサイズのレジ袋の容量(L:liter)
# output エコバッグの使用済み容量、S,Lの枚数のリスト
# eco_bag_capa_used:エコバッグの使用済み容量(%)

def calc_bags_num(item_capa, eco_bag_capa=0, s_bag_capa=15, l_bag_capa=30):

    eco_bag_capa_used = 0
    s_bag_num = 0
    l_bag_num = 0

    if eco_bag_capa > 0:
        eco_bag_capa_used = (int)(item_capa / eco_bag_capa * 100)
        item_capa -= eco_bag_capa

    while item_capa > 0:
        if item_capa > s_bag_capa:
            item_capa -= l_bag_capa
            l_bag_num += 1
        else:
            item_capa -= s_bag_capa
            s_bag_num += 1

    return (eco_bag_capa_used, s_bag_num, l_bag_num)


if __name__ == "__main__":
    bags_num = calc_bags_num(95, 50)
    print("bags_num:\n(エコバッグ使用済み容量(%), S枚数, L枚数) = ", bags_num)
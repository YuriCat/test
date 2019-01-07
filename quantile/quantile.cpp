#include <iostream>
#include <fstream>
#include <utility>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>
#include <random>
#include <limits>

using namespace std;

const int MAX = 10000;
const int samples = 1000000;
const int queries_per_sample = 1;

template <class T>
ostream& operator <<(ostream& ost, const vector<T>& src) {
    ost << "{";
    size_t cnt = 0;
    for (T v : src) {
        ost << v;
        if (++cnt < src.size()) ost << ", ";
    }
    ost << "}";
    return ost;
}

pair<int, double> pos_rest(int count, double p)
{
    if (count == 1) return make_pair(0, 0.0);
    double position = p * (count - 1);
    int pos = min(int(position), count - 2);
    double rest = position - pos;
    return make_pair(pos, rest);
}
template <typename T>
double interpolate(T value0, T value1, double d)
{
    return value0 * (1 - d) + value1 * d;
}

template <typename T>
struct NaiveSort
{
    vector<T> data;
    bool sorted = false;
    void insert(T value) {
        data.push_back(value);
        sorted = false;
    }
    double quantile(double p)
    {
        double position = p * (data.size() - 1);
        int n = min(int(position), (int)data.size() - 2);
        double rest = position - n;
        if (!sorted) sort(data.begin(), data.end());
        sorted = true;
        return data[n] * (1 - rest) + data[n + 1] * rest;
    }
    void show() const
    {}
};

template <typename T>
struct NaiveQuickSelect
{
    vector<T> data;
    void insert(T value) {
        data.push_back(value);
    }
    double quantile(double p)
    {
        double position = p * (data.size() - 1);
        int n = min(int(position), (int)data.size() - 2);
        double rest = position - n;
        nth_element(data.begin(), data.begin() + n, data.end());
        nth_element(data.begin() + n + 1, data.begin() + n + 1, data.end());
        return data[n] * (1 - rest) + data[n + 1] * rest;
    }
    void show() const
    {}
};

template <typename T>
struct NaiveQuantileSetMine
{
    vector<T> data;
    void insert(T value) {
        data.push_back(value);
    }
    double quantile(double p)
    {
        int count = data.size();
        if (count == 0) exit(1);
        if (count == 1) return data[0]; 
        // データ数-1 の区間がある
        double position = p * (count - 1);
        // positionを含む区間の両端のデータのインデックスが
        // pos, pos+1 になるように
        // p=1 の時に備えて データ数-2 を上限とする
        int pos = min(int(position), count - 2);
        // 余り
        double rest = position - pos;
        // データのソート
        //if (!sorted) sort(data.begin(), data.end());
        //sorted = true;
        //T value0 = data[pos];
        //T value1 = data[pos + 1];
        //find_kth_element(pos);
        //find_kth_element(pos + 1, pos + 1);

        find_two_elements(pos, pos + 1);
        // rest : (1 - rest) の内分点を返す
        return data[pos] * (1 - rest) + data[pos + 1] * rest;
    }
    /*T* partition(T *first, T *last, T pivot) {
        cerr << "partition " << data << " " << pivot << endl;
        while (true) {
            while (*first < pivot) ++first;
            --last;
            while (pivot < *last) --last;
            if (!(first < last)) return first;
            swap(first, last);
            ++first;
        }
        cerr << "after " << data << endl;
    }
    int partition_simple(T pivot, int st, int ed)
    {
        int i = st, j = ed - 2;
        while (1)
        {
            while (data[i] < pivot) i++;
            while (data[j] >= pivot) j--;
            if (i >= j) break;
            swap(data[i], data[j]);
        }
        return i;
    }
    pair<int, int> partition(T pivot, int st, int ed)
    {
        int i = st, j = ed - 1;
        int samei = i, samej = j;
        cerr << "partition " << data << " " << pivot << endl;
        while (1)
        {
            while (1)
            {
                cerr << "i = " << i << endl;
                if (data[i] > pivot) break;
                else if (!(data[i] < pivot))
                {
                    if (i < j) {
                        cerr << "same swap!" << endl;
                        swap(data[samei++], data[i]);
                    }
                    else break;
                }
                i++;
            }
            while (1)
            {
                cerr << "j = " << j << endl;
                if (data[j] < pivot) break;
                else if (!(data[j] > pivot))
                {
                    if (i < j) swap(data[samej--], data[j]);
                    else break;
                }
                j--;
            }
            if (i >= j) break;
            swap(data[i], data[j]);
        }
        cerr << "tochu " << data << endl;
        // 同じ値を中心に返していく
        for (int ii = st; ii < samei; ii++) swap(data[ii], data[i - 1 - (ii - st)]);
        for (int jj = ed - 1; jj > samej; jj--) swap(data[jj], data[i + (ed - 1 - jj)]);
        cerr << "after " << data << endl;
        return make_pair(i - (samei - st), i + (ed - 1 - samej));
    }
    static inline T median3(T a, T b, T c)
    {
        return max(min(a,b), min(max(a,b),c));
    }
    void find_kth_element(int pos, int st = 0, int ed = -1)
    {
        if (ed == -1) ed = data.size();
        T *base = data.data();
        T *first = base + st, *last = base + ed;
        T *dst = base + pos;
        while (first + 1 < last)
        {
            T pivot = median3(*first, *(first + (last - first) / 2), *(last - 1));
            cerr << (first-base) << " " << ((first + (last - first) / 2)-base) << " " << ((last - 1)-base) << endl;
            cerr << *first << " " << *(first + (last - first) / 2) << " " << *(last - 1) << endl;
            T *b = partition(first, last, pivot);
            if (dst < b) last = b;
            else if (b < dst) first = b;
            //auto boundaries = partition(pivot, st, ed);
            //cerr << boundaries.first << "," << boundaries.second << endl;
            //if (pos < boundaries.first) ed = boundaries.first;
            //else if (pos >= boundaries.second) st = boundaries.second;
            //else break;
        }
    }*/
    int partition_simple(T pivot, int st, int ed)
    {
        int i = st, j = ed - 2;
        while (1)
        {
            while (data[i] < pivot) i++;
            while (data[j] >= pivot) j--;
            if (i >= j) break;
            swap(data[i], data[j]);
        }
        return i;
    }
    void find_two_elements(int pos0, int pos1)
    {
        int st = 0, ed = data.size();
        while (st + 2 < ed)
        {
            int pivot_index = data[st] > data[(st + ed) / 2] ? st : (st + ed) / 2;
            T pivot = data[pivot_index];
            swap(data[pivot_index], data[ed - 1]);
            int boundary = partition_simple(pivot, st, ed);
            swap(data[boundary], data[ed - 1]);
            if (pos1 < boundary) ed = boundary;   
            else if (pos0 > boundary) st = boundary + 1;
            else
            {
                if (pos0 == boundary) find_kth_element(pos1, boundary + 1, ed);
                else if (pos1 == boundary) find_kth_element(pos0, st, boundary);
                else
                {
                    find_kth_element(pos0, st, boundary);
                    find_kth_element(pos1, boundary + 1, ed);
                }
                return;
            }
        }
        if (data[st] > data[ed - 1]) swap(data[st], data[ed - 1]);
    }
    void find_kth_element(int pos, int st, int ed)
    {
        if (st + 1 >= ed) return;
        int pivot_index = data[st] > data[(st + ed) / 2] ? st : (st + ed) / 2;
        T pivot = data[pivot_index];
        swap(data[pivot_index], data[ed - 1]);
        int boundary = partition_simple(pivot, st, ed);
        swap(data[boundary], data[ed - 1]);
        if (pos < boundary) find_kth_element(pos, st, boundary);
        else if (pos > boundary) find_kth_element(pos, boundary + 1, ed);
    }
    void show() const
    {}
};

template <typename T>
struct NaiveQuantileSetManyQuery
{
    vector<T> data;
    NaiveQuantileSetManyQuery()
    {
        data.push_back(numeric_limits<T>::lowest());
    }
    void insert(T value)
    {
        data.push_back(T(0));
        int index = (int)data.size() - 1;
        while (value < data[index - 1])
        {
            data[index] = data[index - 1];
            index--;
        }
        data[index] = value;
    }
    double quantile(double p) const
    {
        int count = data.size() - 1;
        if (count == 0) exit(1);
        if (count == 1) return data[1]; 
        auto pr = pos_rest(count, p);
        return interpolate(data[pr.first + 1], data[pr.first + 2], pr.second);
    }
    void show() const
    {};
};

/*template <typename T>
struct BoundedNaiveQuantileSetManyQuery
{
    vector<pair<int, T>> data;

    NaiveQuantileSetManyQuery()
    {
        data.push_back(make_pair(0, numeric_limits<T>::lowest()));
    }
    void insert(T value)
    {
        int min, max;
        int index;
        while (min < max)
        {
            index = (min + max) / 2;
            if (value < data[index].second)
            {
                max = index;
            }
            else
            {
                min = index;
            }
        }
        if (data[index].second == value)
        {
            for (int i = index; i < (int)data.size(); i++) data[index].first++;
        }
        else
        {
            data.push_back(make_pair(0, T(0)));
            while (value < data[index - 1].second)
            {
                data[index].first = data[index - 1].first + 1;
                data[index].second = data[index - 1].second;
                index--;
            }
            data[index] = make_pair(data[index - 1] + 1, value);
        }
    }
    int binary_search()
    {

    }
    double quantile(double p) const
    {
        auto pr = pos_rest(data.size() - 1, p);
        auto itr0 = lower_bound(data.begin(), data.end(), pos);
        auto itr1 = lower_bound(itr0, data.end(), pos + 1);

        int min, max;
        int index;
        while (min < max)
        {
            index = (min + max) / 2;
            if (value < data[index].second)
            {
                max = index;
            }
            else
            {
                min = index;
            }
        }

        T value0 = itr0->second;
        T value1 = itr1->second;
        return interpolate(value0, value1, pr.second);
    }
    void show() const
    {};
};*/

template <typename T, int M>
struct BoundedQuantileSet
{
    array<int, M> histgram;
    int count;

    BoundedQuantileSet()
    {
        histgram.fill(0);
        count = 0;
    }
    void insert(T value)
    {
        histgram[value]++;
        count++;
    }
    double quantile(double p) const
    {
        if (count == 0) exit(1);
        auto pr = pos_rest(count, p);
        int pos = pr.first;
        int sum = 0;
        T value0, value1;
        if (p <= 0.5)
        {
            int i = 0;
            while (1)
            {
                sum += histgram[i];
                if (sum > pos) break;
                i++;
            }
            value0 = i;
            if (count == 1) return value0;
            if (sum > pos + 1)
            {
                return value0; // lower == upper のため
            }
            i++;
            while (1)
            {
                sum += histgram[i];
                if (sum > pos + 1) break;
                i++;
            }
            value1 = i;
        }
        else
        {
            pos = count - 1 - (pos + 1);
            int i = M - 1;
            while (1)
            {
                sum += histgram[i];
                if (sum > pos) break;
                i--;
            }
            value1 = i;
            if (count == 1) return value1;
            if (sum > pos + 1)
            {
                return value1; // lower == upper のため
            }
            i--;
            while (1)
            {
                sum += histgram[i];
                if (sum > pos + 1) break;
                i--;
            }
            value0 = i;
        }
        return interpolate(value0, value1, pr.second);
    }
    void show() const
    {};
};

template <typename T, int M>
struct BoundedQuantileSetManyQuery
{
    array<int, M> cum_histgram;

    BoundedQuantileSetManyQuery()
    {
        cum_histgram.fill(0);
    }
    int insert(T value)
    {
        // v以下の値の個数を更新
        for (T v = value; v < M; v++)
        {
            cum_histgram[v]++;
        }
        return cum_histgram[value];
    }
    double quantile(double p) const
    {
        int count = cum_histgram[M - 1];
        if (count == 0) exit(1);
        auto pr = pos_rest(count, p);
        int pos = pr.first + 1;
        auto itr0 = lower_bound(cum_histgram.begin(), cum_histgram.end(), pos);
        T value0 = distance(cum_histgram.begin(), itr0);
        if (count == 1) return value0;
        auto itr1 = lower_bound(itr0, cum_histgram.end(), pos + 1);
        T value1 = distance(cum_histgram.begin(), itr1);
        return interpolate(value0, value1, pr.second);
    }
    void show() const
    {}
};

template <typename T>
struct QuantileSet
{
    struct Node
    {
        T value;
        int count;
        int left_descendant_count;
        int left_child, right_child;
    };
    vector<Node> tree;
    int count;

    QuantileSet() {
        count = 0;
    }

    int insert(T value)
    {
        count++;
        int pos = 0;
        if (!tree.empty()) {
            int index = 0;
            while (1) {
                Node& node = tree[index];
                int next_index;
                if (value < node.value)
                {
                    node.left_descendant_count++;
                    index = node.left_child;
                    if (index == -1)
                    {
                        node.left_child = tree.size();
                        break;
                    }
                }
                else if (value > node.value)
                {
                    pos += node.left_descendant_count + node.count;
                    index = node.right_child;
                    if (index == -1)
                    {
                        node.right_child = tree.size();
                        break;
                    }
                }
                else
                {
                    node.count++;
                    return pos + node.left_descendant_count;
                }
            }
        }
        Node next;
        next.value = value;
        next.count = 1;
        next.left_descendant_count = 0;
        next.left_child = -1;
        next.right_child = -1;
        tree.push_back(next);
        return pos;
    }
    double quantile(double p) {
        if (count == 0) exit(1);
        if (tree.size() == 1) return tree[0].value;
        auto pr = pos_rest(count, p);
        //T value0 = find_kth_element(pr.first);
        //T value1 = find_kth_element(pr.first + 1);
        //return interpolate(value0, value1, pr.second);
        auto values = find_two_elements(pr.first, pr.first + 1);
        return interpolate(values.first, values.second, pr.second);
    }
    T find_kth_element(int pos, int index = 0)
    {
        if (index >= tree.size()) return T(-1);
        while (1) {
            if (index < 0) exit(1);
            const Node& node = tree[index];
            if (pos < node.left_descendant_count)
            {
                index = node.left_child;
            }
            else if (pos >= node.left_descendant_count + node.count)
            {
                index = node.right_child;
                pos -= node.left_descendant_count + node.count;
            }
            else
            {
                return node.value;
            }
        }
        assert(0);
    }
    pair<T, T> find_two_elements(int pos0, int pos1, int index = 0)
    {
        if (index >= tree.size()) return make_pair(T(-1), T(-1));
        T value0, value1;
        while (1) {
            const Node& node = tree[index];
            if (pos0 < node.left_descendant_count)
            {
                if (pos1 < node.left_descendant_count)
                {
                    index = node.left_child;
                }
                else
                {
                    value0 = find_kth_element(pos0, node.left_child);
                    break;
                }
            }
            else
            {
                if (pos0 >= node.left_descendant_count + node.count)
                {
                    index = node.right_child;
                    pos0 -= node.left_descendant_count + node.count;
                    pos1 -= node.left_descendant_count + node.count;
                }
                else
                {
                    value0 = node.value;
                    break;
                }
            }
        }
        const Node& node = tree[index];
        if (pos1 >= node.left_descendant_count + node.count)
        {
            value1 = find_kth_element(pos1 - (node.left_descendant_count + node.count),
                                      node.right_child);
        }
        else
        {
            value1 = node.value;
        }
        return make_pair(value0, value1);
    }
    void show() const
    {
        for (int i = 0; i < (int)tree.size(); i++)
        {
            const Node& node = tree[i];
            cerr << i << " " << node.value << " " << node.count;
            cerr << " " << node.left_child << " " << node.right_child << endl;
        }
    }
};

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

struct PSquare
{
    vector<double> q_;
    vector<int> n_;
    vector<double> n_prime_;
    vector<double> d_n_prime_;

    PSquare(double mean = 0, double quad = 1, int size = 5) {
        clear();
        for (int i = 0; i < size; i++)
        {
            // 論文の実装では 0, p, p/2, (1+p)/2, 1の5点を利用
            // ↑これは絶対なのだろうか
            q_.push_back(mean + quad * (i - double(size - 1) / 2) * 4 / (size - 1));
            n_.push_back(i);
            n_prime_.push_back(i);
            d_n_prime_.push_back(i / double(size - 1));           
        }
    }
    void clear()
    {
        q_.clear();
        n_.clear();
        n_prime_.clear();
        d_n_prime_.clear();
    }
    double parabolic(int i, int d)
    {
        return q_[i] + d / double(n_[i + 1] - n_[i - 1])
        * ((n_[i] - n_[i - 1] + d) * (q_[i + 1] - q_[i]) / (n_[i + 1] - n_[i])
           + (n_[i + 1] - n_[i] - d) * (q_[i] - q_[i - 1]) / (n_[i] - n_[i - 1]));
    }
    double linear(int i, int d)
    {
        return q_[i] + d * (q_[i + d] - q_[i]) / double(n_[i + d] - n_[i]);
    }
    void insert(double val)
    {
        int k = q_.size() - 2;
        if (val < q_.front())
        {
            k = 0;
            q_.front() = val;
        }
        else if (val >= q_.back())
        {
            k = q_.size();
            q_.back() = val;
        }
        else
        {
            for (int i = 1; i < (int)q_.size(); i++)
            {
                if (val < q_[i])
                {
                    k = i;
                    break;
                }
            }
        }
        for (int i = k; i < (int)q_.size(); i++)
        {
            n_[i]++;
        }
        for (int i = 0; i < (int)q_.size(); i++)
        {
            n_prime_[i] += d_n_prime_[i];
        }
        for (int i = 1; i < (int)q_.size() - 1; i++)
        {
            double d = n_prime_[i] - n_[i];
            if ((d >= 1 && n_[i + 1] - n_[i] > 1)
                || (d <= -1 && n_[i - 1] - n_[i] < -1))
            {
                double q_prime = parabolic(i, sgn(d));
                if (q_[i - 1] < q_prime && q_prime < q_[i + 1])
                {
                    q_[i] = q_prime;
                }
                else
                {
                    q_[i] = linear(i, sgn(d));
                }
                n_[i] = n_[i] + sgn(d);
            }
        }
    }
    double quantile(double p) const
    {
        auto pr = pos_rest(q_.size(), p);
        return interpolate(q_[pr.first], q_[pr.first + 1], pr.second);
    }
    void show() const
    {
        //cerr << q_ << n_ << n_prime_ << endl;
    }
};

template <class qset_t>
void test()
{
    qset_t qs;
    qs.insert(1);
    qs.insert(6);
    qs.insert(9);
    qs.insert(0);

    for (float a = 0; a <= 1.01; a += 0.1)
    {
        cerr << a << " " << qs.quantile(a) << endl;
    }
    qs.show();
}

template <class qset_t>
double profile(const vector<int>& rseq, const vector<double>& qrseq)
{
    clock_t start = clock();
    qset_t qs;
    double sum = 0;
    int dcnt = 0;
    for (int r : rseq)
    {
        qs.insert(r);
        for (int i = 0; i < queries_per_sample; i++) sum += qs.quantile(qrseq[dcnt++]);
    }
    double tm = (clock() - start) / double(CLOCKS_PER_SEC);
    cerr << tm << " sec  sum = " << sum << endl;
    return tm;
}

int main()
{
    test<NaiveQuickSelect<int>>();
    test<NaiveQuantileSetMine<int>>();
    test<NaiveQuantileSetManyQuery<int>>();
    test<BoundedQuantileSet<int, MAX>>();
    test<BoundedQuantileSetManyQuery<int, MAX>>();
    test<QuantileSet<int>>();
    test<PSquare>();

    mt19937 mt(0);

    PSquare p2(50, 20);
    uniform_real_distribution<double> rd(0, 100);
    
    ofstream ofs("out.csv");
    for (int i = 0; i < 1000; i++)
    {
        double r = rd(mt);
        p2.insert(r);
        cerr << r << " ->" << p2.q_ << endl;
        for (double d : p2.q_) ofs << d << ",";
        ofs << endl;
        p2.show();
    }

   

    vector<double> tm(10, 0);
    for (int s = 0; s < 10; s++)
    {
        vector<int> brseq;
        vector<double> qrseq;
        uniform_int_distribution<int> uid(0, MAX - 1);
        for (int i = 0; i < samples; i++) brseq.push_back(uid(mt));
        uniform_real_distribution<double> urd(0, 1);
        for (int i = 0; i < queries_per_sample * samples; i++) qrseq.push_back(urd(mt));

        //tm[0] += profile<NaiveSort<int>>(brseq, qrseq);
        //tm[1] += profile<NaiveQuickSelect<int>>(brseq, qrseq);
        //tm[2] += profile<NaiveQuantileSetMine<int>>(brseq, qrseq);
        //tm[3] += profile<NaiveQuantileSetManyQuery<int>>(brseq, qrseq);
        tm[4] += profile<BoundedQuantileSet<int, MAX>>(brseq, qrseq);
        tm[5] += profile<BoundedQuantileSetManyQuery<int, MAX>>(brseq, qrseq);
        tm[6] += profile<QuantileSet<int>>(brseq, qrseq);
        tm[7] += profile<PSquare>(brseq, qrseq);
    }
    vector<double> tm_mean;
    for (double t : tm) tm_mean.push_back(t / 10);
    cerr << tm_mean << endl;
}

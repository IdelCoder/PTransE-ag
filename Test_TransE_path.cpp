#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<tuple>
#include<fstream>

using namespace std;

bool debug=false;

string used = "1";

bool L1_flag=1;

map<pair<string,int>,double>  path_confidence;

vector<int> rel_type;

string version;
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

vector<vector<pair<int,int> > > e1_e3;


// 结点度数列表
map<int, int> entity_degree;

int relation_num,entity_num;
int n= 100;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%9==4)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;

	map<pair<int,int>,vector<pair<vector<int>,double> > >fb_path;
	
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
		// 保存所有的三元组(训练集，验证集)
        ok[make_pair(x,z)][y]=1;
    }

    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
		if (z!=-1)
		{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
			// 保存测试数据的三元组
        	ok[make_pair(x,z)][y]=1;
		}
		if (path_list.size()>0)
		// 保存所有路径信息
		fb_path[make_pair(x,y)] = path_list;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e1][ii]-entity_vec[e2][ii]-relation_vec[rel+relation_num][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e1][ii]-entity_vec[e2][ii]-relation_vec[rel+relation_num][ii]);
		int h = e1;
		int l = e2;
		//used = "0";
		if (used=="1")
		{
			//cout<<"Here"<<endl;
			vector<pair<vector<int>,double> > path_list = fb_path[make_pair(h,l)];
			if (path_list.size()>0)
			{
				for (int path_id = 0; path_id<path_list.size(); path_id++)
				{
					vector<int> rel_path = path_list[path_id].first;

					double pr_path = 0;
					double pr = path_list[path_id].second;
					string  s;
				    ostringstream oss;//创建一个流
					for (int ii=0; ii<rel_path.size(); ii++)
					{
						oss<<rel_path[ii]<<" ";
					}
				    s=oss.str();//
					if (path_confidence.count(make_pair(s,rel))>0)
						pr_path = path_confidence[make_pair(s,rel)];
					//pr_path = 1;
					sum+=calc_path(rel,rel_path)*pr*pr_path;
				}
			}
			path_list = fb_path[make_pair(l,h)];
			if (path_list.size()>0)
			{
				for (int path_id = 0; path_id<path_list.size(); path_id++)
				{
					vector<int> rel_path = path_list[path_id].first;
					double pr = path_list[path_id].second;
					double pr_path = 0;
					string  s;
				    ostringstream oss;//创建一个流
					for (int ii=0; ii<rel_path.size(); ii++)
					{
						oss<<rel_path[ii]<<" ";
					}
				    s=oss.str();//
					if (path_confidence.count(make_pair(s,rel+relation_num))>0)
						pr_path = path_confidence[make_pair(s,rel+relation_num)];
					//pr_path = 1;
					sum+=calc_path(rel+relation_num,rel_path)*pr*pr_path;
				}
			}
		}
        return sum;
    }
    double calc_path(int r1,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];
	        if (L1_flag)
				sum+=-fabs(tmp);
			else
				sum+=-sqr(tmp);
		}
        return 10+sum;
    }
    void run()
    {
		// 读取关系和实体的向量
        FILE* f1 = fopen(("relation2vec.txt"+version).c_str(),"r");
        FILE* f3 = fopen(("entity2vec.txt"+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb*2);
        for (int i=0; i<relation_num_fb*2;i++)
        {
        //	int tmp;
        //	fscanf(f1,"%d",&tmp);
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
        //	int tmp;
        	//fscanf(f3,"%d",&tmp);
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);
		
		
		
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double mid_sum = 0,mid_sum_filter=0;
		double lp_n=0,lp_n_filter = 0;
		double rp_n=0,rp_n_filter = 0;
		double mid_p_n=0,mid_p_n_filter = 0;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> mid_sum_r,mid_sum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,double> mid_p_n_r,mid_p_n_filter_r;
		map<int,int> rel_num;

		// 各度结点的头实体预测表现
		// map<度数, tuple<累计排名，累计Hits@N数，总数> >
		map<int, tuple<long, int, int> > hd_raw;
		map<int, tuple<long, int, int> > td_raw;
		map<int, tuple<long, int, int> > hd_filter;
		map<int, tuple<long, int, int> > td_filter;
		
		
		double l_one2one=0,r_one2one=0,one2one_num=0;
        double l_n2one=0,r_n2one=0,n2one_num=0;
        double l_one2n=0,r_one2n=0,one2n_num=0;
        double l_n2n=0,r_n2n=0,n2n_num=0;
		

		int hit_n = 1;
		// e1_e2是top500的实体对（这是测试结果）
		//FILE* f_e1_e2 = fopen("../e1_e2.txt","w");
		map<pair<int,int>,int> e1_e2;
		for (int testid = 0; testid<fb_l.size()/2; testid+=1)
		{
			int h = fb_h[testid*2];
			int l = fb_l[testid*2];
			int rel = fb_r[testid*2];
			rel_num[rel]+=1;
			// 存储候选实体得分
			vector<pair<int,double> > a;
			
            if (rel_type[rel]==0)
                one2one_num+=1;
            else
            if (rel_type[rel]==1)
                n2one_num+=1;
            else
            if (rel_type[rel]==2)
                one2n_num+=1;
            else
                n2n_num+=1;
			

			double ttt=0;
			int filter = 0;
			// 头实体预测  
			used = "0";
			for (int i=0; i<entity_num; i++)
			{
				// 计算损失值（使用TransE的能量函数）(head prediction)
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}

			int rerank_num = 500;
			// 递减排序
			sort(a.begin(),a.end(),cmp);
			used = "1";
			for (int i=a.size()-1; i>=a.size()-rerank_num; i--)
			{
				// 计算损失值（PTransE的能量函数）(head prediction)
				double sum = calc_sum(a[i].first,l,rel);
				a[i].second = sum;
			}
			
			sort(a.begin(),a.end(),cmp);
			for (int i=a.size()-1; i>=0; i--)
			{
				if (a.size()-i<=rerank_num)
				    //保存top 500的实体，实际没啥用
					e1_e2[make_pair(a[i].first,l)] = 1;
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
				    // 无效的变量
					ttt++;
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	// filter rank
					filter+=1;
				
				if (a[i].first ==h)
				{
					// 当前答案是正确答案
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					// 作用未知
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;

					// 答案的度
					int ans_degree = entity_degree[h];
					if(hd_raw.count(ans_degree) == 0) {
						auto tup_raw = make_tuple(a.size()-i, 0, 1);
						auto tup_filter = make_tuple(filter+1, 0, 1);
						hd_raw[ans_degree] = tup_raw;
						hd_filter[ans_degree] = tup_filter;
					} else {
						get<0>(hd_raw[ans_degree]) += a.size()-i;
						get<0>(hd_filter[ans_degree]) += filter+1;
						get<2>(hd_raw[ans_degree]) ++;
						get<2>(hd_filter[ans_degree])++;
					}
					if (a.size()-i<=hit_n)
					{
						// 排名小于hit_n
						lp_n+=1;
						lp_n_r[rel]+=1;
						get<1>(hd_raw[ans_degree]) ++;
					}
					if (filter<hit_n)
					{
						get<1>(hd_filter[ans_degree]) ++;
						// 关系类型细分的情况
						// 考虑的是filter过后的
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
						if (rel_type[rel]==0)
                            l_one2one+=1;
                        else
                        if (rel_type[rel]==1)
                            l_n2one+=1;
                        else
                        if (rel_type[rel]==2)
                            l_one2n+=1;
                        else
                            l_n2n+=1;
						
					}
					//if (a.size()-i>20)
					//	break;
				}
			}
			a.clear();
			// tail prediction
			used = "0";
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			used = "1";
			sort(a.begin(),a.end(),cmp);
			for (int i=a.size()-1; i>=a.size()-rerank_num; i--)
			{
				double sum = calc_sum(h,a[i].first,rel);
				a[i].second = sum;
			}

			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (a.size()-i<=rerank_num)
					e1_e2[make_pair(h,a[i].first)] = 1;
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					// 答案的度
					int ans_degree = entity_degree[l];
					if(td_raw.count(ans_degree) == 0) {
						auto tup_raw = make_tuple(a.size()-i, 0, 1);
						auto tup_filter = make_tuple(filter+1, 0, 1);
						td_raw[ans_degree] = tup_raw;
						td_filter[ans_degree] = tup_filter;
					} else {
						get<0>(td_raw[ans_degree]) += a.size()-i;
						get<0>(td_filter[ans_degree]) += filter+1;
						get<2>(td_raw[ans_degree]) ++;
						get<2>(td_filter[ans_degree])++;
					}
					if (a.size()-i<=hit_n)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
						get<1>(td_raw[ans_degree])++;
					}
					if (filter<hit_n)
					{
						get<1>(td_filter[ans_degree])++;
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
						;
												if (rel_type[rel]==0)
						                            r_one2one+=1;
						                        else
						                        if (rel_type[rel]==1)
						                            r_n2one+=1;
						                        else
						                        if (rel_type[rel]==2)
						                            r_one2n+=1;
						                        else
						                            r_n2n+=1;
						
					}
					//if (a.size()-i>20)
					//	break;
				}
			}
			a.clear();
			// relation prediction
			for (int i=0; i<relation_num; i++)
			{
				double sum = 0;
				sum+=calc_sum(h,l,i);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,a[i].first)].count(l)>0)
					ttt++;
				if (ok[make_pair(h,a[i].first)].count(l)==0)
			    	filter+=1;
				if (a[i].first==rel)
				{
					mid_sum+=a.size()-i;
					mid_sum_filter+=filter+1;
					mid_sum_r[rel]+=a.size()-i;
					mid_sum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						mid_p_n+=1;
						mid_p_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						mid_p_n_filter+=1;
						mid_p_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			if (testid%100==0)
			{
				// testid head预测平均排名 head预测hits@n平均值 tail预测平均排名 tail预测hits@n平均值
				// filter版本对应的结果
				cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '
				<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"
				<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '
				<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
				// relation预测平均排名 relation预测平均hits （filter）relation预测平均排名 （filter）relation预测平均hits
				cout<<"\t"<<mid_sum/(testid+1)<<' '<<mid_p_n/(testid+1)<<"\t"<<mid_sum_filter/(testid+1)<<' '<<mid_p_n_filter/(testid+1)<<endl;
				// head预测的关系类型数量
               cout<<l_one2one<<" "<<l_one2n<<" "<<l_n2one<<' '<<l_n2n<<endl;
			   // tail预测的关系类型数量
               cout<<r_one2one<<" "<<r_one2n<<" "<<r_n2one<<' '<<r_n2n<<endl;
			   // 各个类型的关系数量
               cout<<one2one_num<<" "<<one2n_num<<" "<<n2one_num<<' '<<n2n_num<<endl;
			   // head prediction和tail prediction的各类细分关系的hits@n
               cout<<l_one2one/one2one_num<<" "<<l_one2n/one2n_num<<" "<<l_n2one/n2one_num<<' '<<l_n2n/n2n_num<<endl;
               cout<<r_one2one/one2one_num<<" "<<r_one2n/one2n_num<<" "<<r_n2one/n2one_num<<' '<<r_n2n/n2n_num<<endl;
				
			}
		}
	//	for (map<pair<int,int>,int>::iterator it = e1_e2.begin(); it!=e1_e2.end(); it++)
	//		fprintf(f_e1_e2,"%d %d\n",it->first.first,it->first.second);
	//	fclose(f_e1_e2);
	//	cout<<"\t"<<mid_sum/(fb_l.size()/2+1)<<' '<<mid_p_n/(fb_l.size()/2+1)<<"\t"<<mid_sum_filter/(fb_l.size()/2+1)<<' '<<mid_p_n_filter/(fb_l.size()/2+1)<<endl;
	//	cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
	//	cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
		/*for (int rel=0; rel<relation_num; rel++)
		{
			int num = rel_num[rel];
			cout<<"rel:"<<id2relation[rel]<<' '<<num<<endl;
			cout<<"left:"<<lsum_r[rel]/num<<'\t'<<lp_n_r[rel]/num<<"\t"<<lsum_filter_r[rel]/num<<'\t'<<lp_n_filter_r[rel]/num<<endl;
			cout<<"right:"<<rsum_r[rel]/num<<'\t'<<rp_n_r[rel]/num<<'\t'<<rsum_filter_r[rel]/num<<'\t'<<rp_n_filter_r[rel]/num<<endl;
			cout<<"mid:"<<mid_sum_r[rel]/num<<'\t'<<mid_p_n_r[rel]/num<<'\t'<<mid_sum_filter_r[rel]/num<<'\t'<<mid_p_n_filter_r[rel]/num<<endl;
		}*/
		// 保存各度的运行结果
		// 分别是hd_raw, hd_filter, td_raw, td_filter
		FILE* f_hd_raw = fopen("hd_raw.csv","w");
		for(auto iter = hd_raw.begin(); iter != hd_raw.end(); iter++) {
			fprintf(f_hd_raw, "%d,%ld,%ld,%d\n", iter->first,get<0>(iter->second),get<1>(iter->second),get<2>(iter->second));
		}
		fclose(f_hd_raw);
		FILE* f_hd_filter = fopen("hd_filter.csv","w");
		for(auto iter = hd_filter.begin(); iter != hd_filter.end(); iter++) {
			fprintf(f_hd_filter, "%d,%ld,%ld,%d\n", iter->first,get<0>(iter->second),get<1>(iter->second),get<2>(iter->second));
		}
		fclose(f_hd_filter);
		FILE* f_td_raw = fopen("td_raw.csv","w");
		for(auto iter = td_raw.begin(); iter != td_raw.end(); iter++) {
			fprintf(f_td_raw, "%d,%ld,%ld,%d\n", iter->first,get<0>(iter->second),get<1>(iter->second),get<2>(iter->second));
		}
		fclose(f_td_raw);
		FILE* f_td_filter = fopen("td_filter.csv","w");
		for(auto iter = td_filter.begin(); iter != td_filter.end(); iter++) {
			fprintf(f_td_filter, "%d,%ld,%ld,%d\n", iter->first,get<0>(iter->second),get<1>(iter->second),get<2>(iter->second));
		}
		fclose(f_td_filter);
    }

};
Test test;

void prepare()
{
	ifstream ip("../../data/degree/estat.csv");
    if(!ip.is_open()) cout << "ERROR: estat not exists" << endl;
    string eName, eId, eDeg;
    while(ip.good()) {
        getline(ip, eName, ',');
        if(eName == "") break;
        getline(ip, eId, ',');
        getline(ip, eDeg, '\n');
        entity_degree[atoi(eId.c_str())] = atoi(eDeg.c_str());
    }

    FILE* f1 = fopen("../../data/entity2id.txt","r");
	FILE* f2 = fopen("../../data/relation2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    FILE* f_kb = fopen("../../data/path2/test_pra.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel;
		fscanf(f_kb,"%d",&rel);
		fscanf(f_kb,"%d",&x);
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_kb,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_kb,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
			fscanf(f_kb,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
		//cout<<e1<<' '<<e2<<' '<<rel<<' '<<b.size()<<endl;
		b.clear();
        test.add(e1,e2,rel,b);
    }
    fclose(f_kb);
	cout<<415<<endl;
    FILE* f_path = fopen("../../data/path2/path2.txt","r");
	while (fscanf(f_path,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_path,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
		fscanf(f_path,"%d",&x);
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_path,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_path,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
			fscanf(f_path,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
		//cout<<e1<<' '<<e2<<' '<<rel<<' '<<b.size()<<endl;
		//b.clear();
        test.add(e1,e2,-1,b);
    }
	cout<<454<<endl;
	fclose(f_path);
    FILE* f_kb1 = fopen("../../data/train.txt","r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen("../../data/valid.txt","r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
	FILE* f_confidence = fopen("../../data/path2/confidence.txt","r");
	while (fscanf(f_confidence,"%d",&x)==1)
	{
		string s = "";
		for (int i=0; i<x; i++)
		{
			fscanf(f_confidence,"%s",buf);
			s = s + string(buf)+" ";
		}
		fscanf(f_confidence,"%d",&x);
		for (int i=0; i<x; i++)
		{
			int y;
			double pr;
			fscanf(f_confidence,"%d%lf",&y,&pr);
		//	cout<<s<<' '<<y<<' '<<pr<<endl;
			path_confidence[make_pair(s,y)] = pr;
		}
	}
	fclose(f_confidence);
    FILE* f7 = fopen("../../data/n2n.txt","r");
    {
        double x,y;
        while (fscanf(f7,"%lf%lf",&x,&y)==2)
        {
            //cout<<x<<' '<<y<<endl;
            if (x<1.5)
            {
                if (y<1.5)
                    rel_type.push_back(0);
                else
                    rel_type.push_back(1);

            }
            else
                if (y<1.5)
                    rel_type.push_back(2);
                else
                    rel_type.push_back(3);
        }
    }
    fclose(f7);
	
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        if (argc==3)
            used = argv[2];
        prepare();
        test.run();
    }
}


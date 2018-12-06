#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<set>
using namespace std;


#define pi 3.1415926535897932384626433832795


map<vector<int>,string> path2s;


map<pair<string,int>,double>  path_confidence;

// 给定一个头结点和关系，得到尾节点，使用下划线将头结点和关系的id分隔开来，作为key
map<string, vector<int> > givenHeadRel;

// 给定一个尾节点和关系，得到头结点
map<string, vector<int> > givenTailRel;

// 给定头结点和尾节点，得到关系
map<string, vector<int> > givenHeadTail;


bool L1_flag=1;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
/*	if (L1_flag)
		for (int i=0; i<a.size(); i++)
			res+=fabs(a[i]);
	else*/
	{
		for (int i=0; i<a.size(); i++)
			res+=a[i]*a[i];
		res = sqrt(res);
	}
	return res;
}

string version;
double positive_margin;
char buf[100000],buf1[100000],buf2[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;




vector<vector<pair<int,int> > > path;

class Train{

public:
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        // 保存该三元组对应的所有路径
		fb_path.push_back(path_list);
        ok[make_pair(x,z)][y]=1;
    }
    void run()
    {
        // n是维数，rate是学习率
        n = 100;
        rate = 0.001;
		cout<<"n="<<n<<' '<<"rate="<<rate<<endl;
        // vector<vector<double> > 二维数组，每一个关系用n为向量表示
		relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
        // 初始化实体和关系的向量
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }

        bfgs();
    }

private:
    int n;
    double res;//loss function value
    int positive_margin_count;
    double count,count1;//loss function gradient
    double rate;//learning rate
    double belta;
    vector<int> fb_h,fb_l,fb_r;
	vector<vector<pair<vector<int>,double> > >fb_path;
    vector<vector<int> > feature;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<double> > relation_tmp,entity_tmp;

    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
        // margin是损失函数L(h,r,t)使用的，margin_rel是损失函数L(p,r)使用的
        double margin = 1,margin_rel = 1;
        cout<<"margin="<<' '<<margin<<"margin_rel="<<margin_rel<<endl;
        cout <<"positive_margin="<< " " << positive_margin <<endl;
        res=0;
        // mini-batch中每一批的样本量
        int nbatches=100;
        // 可能等同于nepoch，迭代的轮数
        int neval = 500;
        int batchsize = fb_h.size()/nbatches;
 		relation_tmp=relation_vec;
		entity_tmp = entity_vec;
		map<string, int> fb_count;
        for (int eval=0; eval<neval; eval++)
        {
            // 每轮迭代
        	res=0;
            positive_margin_count = 0;
         	for (int batch = 0; batch<nbatches; batch++)
         	{
                // 每一批次，都随机选择一个头结点（似乎没有使用）
				int e1 = rand_max(entity_num);
         		for (int k=0; k<batchsize; k++)
         		{
                    // 每个样本
					int j=rand_max(entity_num);
					int i=rand_max(fb_h.size());
                    // fb_h，fb_r，fb_l的索引都是三元组的索引，相同索引，就是同一个三元组
					int e1 = fb_h[i], rel = fb_r[i], e2  = fb_l[i];
                    

					int rand_tmp = rand()%100;
					if (rand_tmp<25)
					{
                        // ok的类型是map<pair<int,int>, map<int,int> >
						while (ok[make_pair(e1,rel)].count(j)>0)
                            j=rand_max(entity_num);
                        train_kb(e1,e2,rel,e1,j,rel,margin);
                        
                        int candidate_size = givenHeadRel[to_string(e1) + "_" + to_string(rel)].size();
                        if(candidate_size > 1) {
                            j = givenHeadRel[to_string(e1) + "_" + to_string(rel)][rand() % candidate_size];
                            while (j == e2) {
                                j = givenHeadRel[to_string(e1) + "_" + to_string(rel)][rand() % candidate_size];
                            }
                           
                            train_kb(e2, j, true, positive_margin);
                        }
                        

                        // 固定头结点和关系，随机出训练集中不存在的三元组来训练


					}
					else
					if (rand_tmp<50)
					{
						while (ok[make_pair(j,rel)].count(e2)>0)
                            j=rand_max(entity_num);
                        train_kb(e1,e2,rel,j,e2,rel,margin);
                        
                        int candidate_size = givenTailRel[to_string(e2) + "_" + to_string(rel)].size();
                        if(candidate_size > 1) {
                            j = givenTailRel[to_string(e2) + "_" + to_string(rel)][rand() % candidate_size];
                            while (j == e1) {
                                j = givenTailRel[to_string(e2) + "_" + to_string(rel)][rand() % candidate_size];
                            }
                        
                    
                            train_kb(e1, j, true, positive_margin);
                        }



					}
					else
					{
                        // rand_tmp [50, 100)
						int rel_neg = rand_max(relation_num);
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
                            rel_neg = rand_max(relation_num);
                        train_kb(e1,e2,rel,e1,e2,rel_neg,margin);

					}
                    // fb_path的类型vector<vector<pair<vector<int>,double> > >
					if (fb_path[i].size()>0)
					{
                        // 这个三元组存在其他路径
                        // relation_num包括了逆关系
						int rel_neg = rand_max(relation_num);
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
                        // 随机出一个不存在的三元组
						for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
						{
                            // 遍历第i个三元组对应的每一条路径
                            // fb_path类型vector<vector<pair<vector<int>,double> > >
                            // rel_path 就是某条路径所组成的关系的数组
							vector<int> rel_path = fb_path[i][path_id].first;
							string  s = "";
                            // path2s的类型是map<vector<int>,string>
							if (path2s.count(rel_path)==0)
							{
							    ostringstream oss;//创建一个流
                                // 将这些数组组合成一个路径的字符串
								for (int ii=0; ii<rel_path.size(); ii++)
								{
									oss<<rel_path[ii]<<" ";
								}
							    s=oss.str();//
                                // path2s的key是路径的关系数组，value是关系数组转成的字符串
								path2s[rel_path] = s;
							}
                            // s是路径的字符串
							s = path2s[rel_path];
                            // pr是每条路径的概率
							double pr = fb_path[i][path_id].second;
                            // pr_path是路径的置信度
							double pr_path = 0;
							if (path_confidence.count(make_pair(s,rel))>0)
								pr_path = path_confidence[make_pair(s,rel)];
							pr_path = 0.99*pr_path + 0.01;
                            // rel是三元组实际存在的关系，rel_neg是这个三元组两个实体间不存在的关系
                            // rel_path是这个关系对应的一个路径
							train_path(rel,rel_neg,rel_path,2*margin,pr*pr_path);
						}
					}
                    // e1,e2,rel组成了这个三元组，j则是用来替换头或者尾的实体，构成一个corrupted triplets
					norm(relation_tmp[rel]);
            		norm(entity_tmp[e1]);
            		norm(entity_tmp[e2]);
            		norm(entity_tmp[j]);
                    // 这句话也没有效果
					e1 = e2;
         		}
                // 每一批迭代过后，更新relation_vec和entity_vec
	            relation_vec = relation_tmp;
	            entity_vec = entity_tmp;
         	}
            // 每一轮迭代过后，记录实体和关系的向量
            cout<<"eval:"<<eval<<' '<<res<<' '<<positive_margin_count<<endl;
            FILE* f2 = fopen(("relation2vec.txt"+version).c_str(),"w");
            FILE* f3 = fopen(("entity2vec.txt"+version).c_str(),"w");
            // 每一行对应一个关系的向量
            for (int i=0; i<relation_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                fprintf(f2,"\n");
            }
            for (int i=0; i<entity_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                fprintf(f3,"\n");
            }
            fclose(f2);
            fclose(f3);
        }//所有epoch结束
    }


    double res1;
    double calc_kb(int e1,int e2,int rel)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
    }

    // 一个三元组合，belta只是为了固定好逆梯度学习的方向
    // 比如，正例使用-1,而corrupted triplet 却是1，这是由求偏导的结果决定的。
    void gradient_kb(int e1,int e2,int rel, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {
            // entity_vec vector<vector<double> >
            double x = 2*(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel][ii]-=belta*rate*x;
            entity_tmp[e1][ii]-=belta*rate*x;
            entity_tmp[e2][ii]+=belta*rate*x;
        }
    }

    // r1是关系，rel_path是拥有关系r1的某个实体对之间的某条路径
    // 计算E(p,r)。 E(h,p,t)=||p-(t-h)||=||p-r||=E(p,r)
    double calc_path(int r1,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = relation_vec[r1][ii];
            // 直连关系-路径里面的每个关系
			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
    }

    // belta是路径的可靠度，belta的符号也是为了逆梯度学习而设置的
    void gradient_path(int r1,vector<int> rel_path, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {

			double x = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				x-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[r1][ii]+=belta*rate*x;
			for (int j=0; j<rel_path.size(); j++)
            	relation_tmp[rel_path[j]][ii]-=belta*rate*x;
        }
    }

    // 前三个构成original triplet,后三个构成corrupted triplet(现在的corrupted triplet还包括corrupt relation)
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double margin)
    {
        double sum1 = calc_kb(e1_a,e2_a,rel_a);
        double sum2 = calc_kb(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
            // 两者的差距在marign之内，才计算损失函数，调整参数
        	res+=margin+sum1-sum2;
        	gradient_kb(e1_a, e2_a, rel_a, -1);
			gradient_kb(e1_b, e2_b, rel_b, 1);
        }
    }
	
	
    void train_kb(int a, int b, bool isEntity, double margin)
    {
	    /*
		 * This method updates the representations of entities/relations when the distance
		 * exceeds the threshold (i.e., margin).
		 * The first two input parameters represent entity pair or relation pair
		 * which are judged by boolean isEntity.
		 */
        double distance = 0;
        for(int ii = 0; ii < n; ii++)
        {
            if(isEntity)
            {
                distance += (entity_vec[a][ii] - entity_vec[b][ii]) * (entity_vec[a][ii] - entity_vec[b][ii]);
            } else
            {
                distance += (relation_vec[a][ii] - relation_vec[b][ii]) * (relation_vec[a][ii] - relation_vec[b][ii]);
            }
        }
        if(sqrt(distance) > margin)
        {
			// Distance exceeds margin, then update embeddings
            positive_margin_count ++;
            for(int ii = 0; ii < n; ii++)
            {
			    // dimension-wise gradient descendant updating
                if(isEntity)
                {
                    entity_tmp[a][ii] -= rate * 2 * (entity_vec[a][ii] - entity_vec[b][ii]);
                    entity_tmp[b][ii] += rate * 2 * (entity_vec[a][ii] - entity_vec[b][ii]);
                } else
                {
                    relation_tmp[a][ii] -= rate * 2 * (relation_vec[a][ii] - relation_vec[b][ii]);
                    relation_tmp[b][ii] += rate * 2 * (relation_vec[a][ii] - relation_vec[b][ii]);
                }
            }
        }

    }
	
	
    // margin = 2*margin, x = pr*pr_path
    void train_path(int rel, int rel_neg, vector<int> rel_path, double margin,double x)
    {
        double sum1 = calc_path(rel,rel_path);
        double sum2 = calc_path(rel_neg,rel_path);
		double lambda = 1;
        if (sum1+margin>sum2)
        {
            // 训练path
            // 记录误差
        	res+=x*lambda*(margin+sum1-sum2);
        	gradient_path(rel,rel_path, -x*lambda);
			gradient_path(rel_neg,rel_path, x*lambda);
        }
    }
};

Train train;
void prepare()
{
    
    FILE* f1 = fopen("./data/entity2id.txt","r");
	FILE* f2 = fopen("./data/relation2id.txt","r");
	int x;
   
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
        // map<string,int> key是实体名，value是实体id。
		entity2id[st]=x;
        // map<int,string> key是id，value是实体名
		id2entity[x]=st;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
        // key是id,value是关系名
		id2relation[x]=st;
        // id加上1345，对应到“-关系名”，1345表示有1345个关系（应该是针对于数据集FB15k的）
		id2relation[x+1345] = "-"+st;
		relation_num++;
	}
    FILE* f_train = fopen("../data/train.txt", "r");
    //cout << "getting train data" << endl;
    while (fscanf(f_train, "%s", buf) == 1) {
         string head = buf;
         //cout << "start: " << buf << endl;
         fscanf(f_train,"%s",buf);
         string tail = buf;
          fscanf(f_train,"%s",buf);
          string rel = buf;
        // 头结点和关系组合
        string headRel = to_string(entity2id[head]) + "_" + to_string(relation2id[rel]);
       // cout << headRel << endl;
        map<string, vector<int> >::iterator iter = givenHeadRel.find(headRel);
        if(iter == givenHeadRel.end()) {
            // key不存在
            vector<int> value = vector<int>();
            value.push_back(entity2id[tail]);
            givenHeadRel.insert(pair<string, vector<int> >(headRel, value));
        } else {
            iter->second.push_back(entity2id[tail]);
        }
        
        // 尾节点和关系组合
        
        string tailRel = to_string(entity2id[tail]) + "_" + to_string(relation2id[rel]);
        //cout << tailRel << endl;
        iter = givenTailRel.find(tailRel);
        if(iter == givenTailRel.end()) {
            // key不存在
            vector<int> value = vector<int>();
            value.push_back(entity2id[head]);
            givenTailRel.insert(pair<string, vector<int> >(tailRel, value));
        } else {
            iter->second.push_back(entity2id[head]);
        }
        
        // 头结点和尾节点组合
        string headTail = to_string(entity2id[head]) + "_" + to_string(entity2id[tail]);
        //cout << headTail << endl;
        iter = givenHeadTail.find(headTail);
        if(iter == givenHeadTail.end()) {
            // key不存在
            vector<int> value = vector<int>();
            value.push_back(relation2id[rel]);
            givenHeadTail.insert(pair<string, vector<int> >(headTail, value));
        } else {
            iter->second.push_back(relation2id[rel]);
        }
        
    }

    FILE* f_kb = fopen("../data/train_pra.txt","r");
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
        // rel是关系id
		fscanf(f_kb,"%d",&rel);
        // 就是实体之间的路径的数目
		fscanf(f_kb,"%d",&x);
        // pair<路径，精度>
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
            // y是某条路径的长度（关系数）
			fscanf(f_kb,"%d",&y);
			for (int j=0; j<y; j++)
			{
                // 读取y所指的路径的关系，
				fscanf(f_kb,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
            // 读取这条路径的精度
			fscanf(f_kb,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
		//cout<<e1<<' '<<e2<<' '<<rel<<' '<<b.size()<<endl;
        // 训练需要的参数：头实体，尾实体，直连关系，（路径加精度）
        train.add(e1,e2,rel,b);
    }
    // 以上就是训练过程
    // 关系数目，包括了逆关系
	relation_num*=2;

    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
	// confidence存放的是每条路径对于每个关系的置信度。比如e1-r1-e2，直连的关系是r1，e3-r2-e4的直连关系是r2。
    // 假设图中只有e1-e2和e3-e4之间存在路径r3-r4。那么，r3-r4只对r1和r2存在置信度，且均是0.5。
	FILE* f_confidence = fopen("./data/confidence.txt","r");
	while (fscanf(f_confidence,"%d",&x)==1)
	{
        // x是路径的长度
		string s = "";
		for (int i=0; i<x; i++)
		{
            // 读取路径中的关系
			fscanf(f_confidence,"%s",buf);
            // 拼接各个关系，组成路径字符串（每个关系是用id表示的）
			s = s + string(buf)+" ";
		}
        // 涉及到的关系数（如上注释的情况，就会是2）
		fscanf(f_confidence,"%d",&x);
		for (int i=0; i<x; i++)
		{
			int y;
			double pr;
            // 读取涉及到的关系以及其置信度
			fscanf(f_confidence,"%d%lf",&y,&pr);
		//	cout<<s<<' '<<y<<' '<<pr<<endl;
            // map<pair<string,int>,double> map<pair<路径字符串，涉及关系id>, 置信度>
			path_confidence[make_pair(s,y)] = pr;
		}
	}
	fclose(f_confidence);
    fclose(f_kb);
}

int main(int argc,char**argv)
{
    //随机数种子
    srand((unsigned) time(NULL));
    if (argc!=2)
        return 0;
    else
    {
        version = argv[1];
        positive_margin = atof(version.c_str());
        prepare();
        train.run();
    }
}

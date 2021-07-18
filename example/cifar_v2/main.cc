#include <iostream>
#include "mlfe/optimizers_v2/sgd.h"
#include "mlfe/optimizers_v2/adam.h"
#include "dataset/cifar.h"
#include "net_models.h"

using namespace mlfe;

// custom metric function.
float categorical_accuracy(Tensor y_true, Tensor y_pred);

void train_convnet(
    dataset::cifar10_gen train_set,
    dataset::cifar10_gen valid_set);

int main(int argc, char *argv[])
{
    try{
        std::vector<uint8_t> train_x;
        std::vector<uint8_t> train_y;
        std::vector<uint8_t> valid_x;
        std::vector<uint8_t> valid_y;

        if(argc < 3)
        {
            std::cout<<argv[0];
            std::cout << " [cifar10]";
            std::cout<<" [cifar dataset folder]"<<std::endl;
            return 1;
        }
        // read all data from original cifar10 binary file.
        dataset::read_cifar10_dataset(argv[2],
            train_x, train_y,
            valid_x, valid_y);

        if (std::string(argv[1]) == "cifar10")
        {
            dataset::cifar10_gen train_set(train_x, train_y), valid_set(valid_x, valid_y);
            train_convnet(train_set, valid_set);
        }
        else
        {
            std::cout<<"Wrong command, ";
            std::cout<<"select one of the commands below."<<std::endl;
            std::cout<<" - classifiar"<<std::endl;
        }
    }
    catch(std::exception &e){
        std::cout<<e.what()<<std::endl;
    }

    return 0;
}

void fill_batch(
    dataset::cifar10_gen & dataset,
    std::vector<float> & x,
    std::vector<float> & y,
    int batch_size, int batch_idx
    )
{
    for (int n = 0; n < batch_size; ++n)
    {
        auto [imgs, labels] = dataset(batch_size * batch_idx + n);
        std::copy(imgs.begin(), imgs.end(), x.begin() + n * imgs.size());
        std::copy(labels.begin(), labels.end(), y.begin() + n * labels.size());
    }
}

void train_convnet(
    dataset::cifar10_gen train_set,
    dataset::cifar10_gen valid_set)
{
    const int BATCH = 128;
    const int EPOCH = 30;
    const int INPUT_SIZE = 32 * 32 * 3;
    const int OUTPUT_SIZE = 10;
    const int NUM_ITER = train_set.size() / BATCH;

    auto model = models::cifar10_convnet();

    auto images = std::vector<float>(BATCH*INPUT_SIZE);
    auto labels = std::vector<float>(BATCH*OUTPUT_SIZE);
    optimizers::adam opt(1e-3);
    opt.set_variables(model.trainable_variables());
    for(int e = 0; e < EPOCH; ++e)
    {
        float train_loss = 0.f;
        float valid_acc = 0.f;
        for(int i = 0; i < NUM_ITER; ++i)
        {
            fill_batch(train_set, images, labels, BATCH, i);
            model.zero_grad();
            auto x = Tensor::from_vector(images, {BATCH, 32, 32, 3});
            auto y_true = Tensor::from_vector(labels, {BATCH, OUTPUT_SIZE});
            auto y_pred = model.forward(x, true);
            auto loss = model.criterion(y_true, y_pred);
            loss.backprop_v2();
            opt.update();
            train_loss += loss.data<float>()[0];
            if((i+1) % 10 == 0){
                std::cout<<(i + 1)<<": "<<train_loss / (i + 1)<<std::endl;
            }
        }
        train_loss /= NUM_ITER;
        const int VALID_ITER = valid_set.size() / BATCH;

        for(int i = 0; i < VALID_ITER; ++i)
        {
            fill_batch(valid_set, images, labels, BATCH, i);
            auto x = Tensor::from_vector(images, {BATCH, 32, 32, 3});
            auto y_true = Tensor::from_vector(labels, {BATCH, OUTPUT_SIZE});
            auto y_pred = model.forward(x);
            valid_acc += categorical_accuracy(y_true, y_pred);
        }
        valid_acc /= VALID_ITER;
        std::cout<<"[EPOCH "<<e + 1<<"] ";
        std::cout<<"train loss: "<<train_loss;
        std::cout<<", valid accuracy: "<<valid_acc<<std::endl;
    }
}


float categorical_accuracy(Tensor y_true, Tensor y_pred)
{
    const int batch_size = y_true.shape()[0];
    const int classes = y_true.shape()[1];
    int correct = 0;
    for(int b = 0; b < batch_size; ++b)
    {
        auto y_pred_pos = std::max_element(
            y_pred.cbegin<float>() + b * classes,
            y_pred.cbegin<float>() + (b + 1) * classes);
        int class_id = std::distance(
            y_pred.cbegin<float>() + b * classes,
            y_pred_pos);
        if(y_true.data<float>()[b * classes + class_id] == 1.f)
        {
            correct += 1;
        }
    }
    return float(correct) / float(batch_size);
}

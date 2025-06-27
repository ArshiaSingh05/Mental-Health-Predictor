import React from "react"
import "./services.css"
import { servicesCard } from "../../dummydata"

const ServicesCard = () => {
  return (
    <>
      <section className='servicesCard'>
        <div className='container grid2'>
          {servicesCard.map((val) => (
            <div className='items'>
              <div className='content flex'>
                <div className='left'>
                  <div className='img'>
                    <img src={val.cover} alt='' />
                  </div>
                </div>
                <div className='text'>
                  <h1>{val.servicesName}</h1>
                  <div className='rate'>
                    <i className='fa fa-star'></i>
                    <i className='fa fa-star'></i>
                    <i className='fa fa-star'></i>
                    <i className='fa fa-star'></i>
                    <i className='fa fa-star'></i>
                    <label htmlFor=''>(5.0)</label>
                  </div>
                </div>
              </div>
              <button className='outline-btn'>TRY IT NOW !</button>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

export default ServicesCard
